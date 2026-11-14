"""AI jobs for the stock tracker, run on the HKBU GenAI API.

Three jobs, each returning checked JSON:

  sort_news      Reads new headlines only. Writes a two sentence summary in its own
                 words for each one, tags impact, importance and topic, drops noise,
                 and groups headlines about the same event into one story.
  write_brief    The 30 second view: a mood label and three sentences on what
                 happened, why, and what it means. Every sentence names its stories.
  write_outlook  Runs once a day or after big news: what happens next, the driver
                 scorecard, and what would confirm the bull or bear case.

Every answer is checked before it is used. If the AI returns something broken,
the job raises AIError and update.py keeps the last good version instead.

Test a model directly:
    python scripts/ai.py gemini-3.8-flash
"""

import json
import os
import re
import sys
from collections import Counter

import requests

HKBU_CHAT_URL = "https://genai.hkbu.edu.hk/api/v1/chat/completions"
TIMEOUT_SECONDS = 180
NEWS_BATCH_SIZE = 10

IMPACTS = ("positive", "negative", "neutral")
IMPORTANCE = ("high", "medium", "low")
TOPICS = ("demand", "competition", "regulation", "leadership", "supply_chain",
          "earnings", "valuation", "macro", "product", "other")
TONES = ("positive", "negative", "neutral")
DIRECTIONS = ("positive", "negative", "mixed")
TIMEFRAMES = ("days", "weeks", "months")
CONFIDENCE = ("low", "medium", "high")
DRIVER_STATUS = ("improving", "steady", "worsening")

HOUSE_RULES = (
    "You are an equity research analyst writing for a busy beginner investor. "
    "Use only the facts in the data you are given. Never invent news, numbers, dates or quotes. "
    "Write in your own words and do not copy phrases from the headlines or summaries, "
    "apart from names and numbers. Use plain English with short sentences. "
    "Do not use dashes as punctuation. Do not give buy, sell or hold advice. "
    "Reply with one JSON object only, no markdown and no extra text."
)


class AIError(Exception):
    """The AI call failed or returned something unusable."""


# ---------------------------------------------------------------------------
# Small checking helpers
# ---------------------------------------------------------------------------

def choice(value, allowed, fallback):
    value = str(value or "").strip().lower().replace(" ", "_")
    return value if value in allowed else fallback


def text(value, limit):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = value.replace(" — ", ", ").replace("—", ", ").replace(" – ", ", ")
    return value[:limit].rstrip()


def id_list(value, known):
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v) in known][:6]


def parse_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise AIError("reply had no JSON object")
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AIError(f"reply JSON could not be read ({exc.msg})") from None


def salvage_items(raw):
    """Rescue the complete {...} entries from a broken or cut off "items" list."""
    raw = raw or ""
    start = raw.find("[", raw.find('"items"') if '"items"' in raw else 0)
    if start == -1:
        return []
    found, depth, begin, in_string, escaped = [], 0, None, False, False
    for index in range(start + 1, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                begin = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and begin is not None:
                try:
                    found.append(json.loads(raw[begin:index + 1]))
                except json.JSONDecodeError:
                    pass
                begin = None
        elif char == "]" and depth == 0:
            break
    return found


def slug(value):
    value = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return value[:48] or "story"


# ---------------------------------------------------------------------------
# The AI class
# ---------------------------------------------------------------------------

class AI:
    def __init__(self, key, fast_model, deep_model):
        self.key = key
        self.fast_model = fast_model
        self.deep_model = deep_model
        self.tokens = Counter()
        self.calls = Counter()
        self.json_mode_off = set()

    def chat(self, model, user_prompt, max_tokens):
        """One call. Returns the parsed JSON object."""
        return parse_json(self.chat_text(model, user_prompt, max_tokens))

    def chat_text(self, model, user_prompt, max_tokens):
        """One call. Returns the raw reply text."""
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": HOUSE_RULES},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if model not in self.json_mode_off:
            body["response_format"] = {"type": "json_object"}

        response = self._post(body)
        if response.status_code == 400 and "response_format" in body:
            # Some models reject JSON mode. Remember that and try once without it.
            self.json_mode_off.add(model)
            body.pop("response_format")
            response = self._post(body)
        if response.status_code >= 400:
            raise AIError(f"HKBU GenAI returned {response.status_code}: {response.text[:200]}")

        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise AIError("HKBU GenAI answer had an unexpected shape") from None

        usage = payload.get("usage") or {}
        self.tokens[model] += int(usage.get("total_tokens") or 0)
        self.calls[model] += 1

        content = message.get("content") or ""
        if not content.strip():
            finish = payload["choices"][0].get("finish_reason")
            raise AIError(f"empty reply (finish reason: {finish}); try a higher max_tokens")
        return content

    def _post(self, body):
        try:
            return requests.post(
                HKBU_CHAT_URL,
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                json=body,
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise AIError(f"network problem reaching HKBU GenAI ({type(exc).__name__})") from None

    # =======================================================================
    # Job 1: sort the news
    # =======================================================================

    def sort_news(self, company, new_items, known_stories, max_tokens=6000):
        """Returns {item_id: {...tags...}} and a dict of new story titles.

        new_items: list of news dicts from sources.py that have not been sorted yet.
        known_stories: {story_key: title} for stories already on file, so new
                       headlines about the same event join the existing story.
        """
        results, new_titles, failures = {}, {}, []
        queue = [new_items[i:i + NEWS_BATCH_SIZE] for i in range(0, len(new_items), NEWS_BATCH_SIZE)]
        while queue:
            batch = queue.pop(0)
            try:
                batch_results, batch_titles = self._sort_batch(company, batch, {**known_stories, **new_titles}, max_tokens)
                results.update(batch_results)
                new_titles.update(batch_titles)
            except AIError as exc:
                if len(batch) > 3:
                    # Try the two halves separately before giving up on this batch.
                    half = len(batch) // 2
                    queue[:0] = [batch[:half], batch[half:]]
                    print(f"news batch of {len(batch)} failed ({exc}); retrying in halves")
                else:
                    failures.append(str(exc))
                    print(f"news batch of {len(batch)} failed: {exc}")
        if not results and failures:
            raise AIError(failures[0])
        return results, new_titles

    def _sort_batch(self, company, batch, known_stories, max_tokens):
        items = [
            {"id": i["id"], "headline": i["headline"], "summary": i.get("summary", "")[:400],
             "source": i.get("source"), "published": i.get("published")}
            for i in batch
        ]
        prompt = (
            f"Company: {company['name']} ({company['ticker']}).\n"
            f"Existing stories (key: title): {json.dumps(known_stories)[:3000]}\n\n"
            f"New headlines:\n{json.dumps(items, indent=1)}\n\n"
            "For every headline, return an entry with these fields:\n"
            '  "id": the headline id, unchanged\n'
            f'  "relevant": true only if it says something real about {company["name"]}, its products, customers, '
            "rivals or industry. Lists like 'stocks to buy' and pieces that only mention the name are false.\n"
            '  "summary": two short sentences in your own words: what happened, and why it matters for the company\n'
            f'  "impact": one of {list(IMPACTS)} for the company\n'
            f'  "importance": one of {list(IMPORTANCE)}. High means it could change the long term story '
            "or move the stock several percent.\n"
            f'  "topic": one of {list(TOPICS)}\n'
            '  "story": the key of an existing story if this headline is about the same event, '
            "otherwise a new short key in lowercase words joined by hyphens. Headlines about the same event "
            "must share one key.\n"
            '  "story_title": for a new story key, a short neutral title of at most 9 words; otherwise ""\n\n'
            'Return {"items": [ ...one entry per headline... ]}'
        )
        raw = self.chat_text(self.fast_model, prompt, max_tokens)
        try:
            rows = parse_json(raw).get("items")
        except AIError:
            rows = None
        if not isinstance(rows, list):
            rows = salvage_items(raw)
            if len(rows) < max(1, len(batch) // 2):
                raise AIError("news reply JSON could not be read")

        known_ids = {i["id"] for i in batch}
        results, titles = {}, {}
        for row in rows:
            if not isinstance(row, dict) or str(row.get("id")) not in known_ids:
                continue
            key = slug(row.get("story"))
            title = text(row.get("story_title"), 90)
            if key not in known_stories and key not in titles:
                titles[key] = title or text(next(i["headline"] for i in batch if i["id"] == str(row["id"])), 90)
            results[str(row["id"])] = {
                "relevant": bool(row.get("relevant", True)),
                "ai_summary": text(row.get("summary"), 420),
                "impact": choice(row.get("impact"), IMPACTS, "neutral"),
                "importance": choice(row.get("importance"), IMPORTANCE, "low"),
                "topic": choice(row.get("topic"), TOPICS, "other"),
                "story": key,
            }
        if not results:
            raise AIError("news reply matched none of the headlines")
        return results, titles

    # =======================================================================
    # Job 2: the 30 second brief
    # =======================================================================

    def write_brief(self, company, facts, stories, max_tokens=2500):
        """facts: numbers from update.py. stories: the top stories with their keys."""
        story_keys = {s["key"] for s in stories}
        prompt = (
            f"Company: {company['name']} ({company['ticker']}).\n"
            f"Market numbers (percent values are already percentages; forward_eps_next_12_months is a blend of "
            f"this fiscal year and next, not a single fiscal year):\n{json.dumps(facts, indent=1)}\n\n"
            f"Top stories from the last few days:\n{json.dumps(stories, indent=1)[:6000]}\n\n"
            "Write the 30 second brief. Return:\n"
            '{"mood": two or three words for how the stock is doing right now, such as "Under pressure", '
            '"Steady", "Strong day", "Recovering",\n'
            f' "tone": one of {list(TONES)},\n'
            ' "headline": one sentence of at most 16 words that sums up the situation,\n'
            ' "sentences": exactly three entries, each {"text": one sentence, "stories": [story keys it relies on]}. '
            "Sentence 1 says what happened to the stock. Sentence 2 says the most likely reason, using the stories. "
            "Sentence 3 says what it means for someone holding the stock for years.}"
        )
        data = self.chat(self.fast_model, prompt, max_tokens)
        sentences = data.get("sentences")
        if not isinstance(sentences, list) or len(sentences) < 2:
            raise AIError("brief reply did not have the sentences")
        clean = []
        for entry in sentences[:3]:
            if isinstance(entry, str):
                entry = {"text": entry, "stories": []}
            if not isinstance(entry, dict) or not text(entry.get("text"), 10):
                continue
            clean.append({"text": text(entry["text"], 300), "stories": id_list(entry.get("stories"), story_keys)})
        if len(clean) < 2:
            raise AIError("brief sentences were empty")
        return {
            "mood": text(data.get("mood"), 28) or "Update",
            "tone": choice(data.get("tone"), TONES, "neutral"),
            "headline": text(data.get("headline"), 140),
            "sentences": clean,
            "model": self.fast_model,
        }

    # =======================================================================
    # Job 3: the outlook
    # =======================================================================

    def write_outlook(self, company, facts, stories, extra, max_tokens=6000):
        """extra: analyst ratings, targets, earnings history, calendar and so on."""
        story_keys = {s["key"] for s in stories}
        drivers = company.get("drivers") or []
        prompt = (
            f"Company: {company['name']} ({company['ticker']}).\n"
            f"Market numbers:\n{json.dumps(facts, indent=1)}\n\n"
            f"Analyst, earnings and calendar data:\n{json.dumps(extra, indent=1)[:6000]}\n\n"
            f"Stories from the last 7 days:\n{json.dumps(stories, indent=1)[:9000]}\n\n"
            f"The investor tracks these drivers: {json.dumps(drivers)}\n\n"
            "Write the outlook: the likely consequences of the current news and what drives the stock next. "
            "This is analysis, not a prediction or advice. Return:\n"
            '{"summary": two or three sentences on where things stand and what matters most next,\n'
            ' "what_next": three to five entries, each {"point": one or two sentences, '
            f'"direction": one of {list(DIRECTIONS)}, "timeframe": one of {list(TIMEFRAMES)}, '
            f'"confidence": one of {list(CONFIDENCE)}, "stories": [story keys it relies on]}},\n'
            ' "drivers": one entry for each driver above, in the same order, each {"name": the driver name, '
            f'"status": one of {list(DRIVER_STATUS)}, "reason": one sentence, "stories": [story keys]}}. '
            'Use "steady" when the data says nothing new about a driver.\n'
            ' "bull_watch": two or three short things that would confirm the optimistic case,\n'
            ' "bear_watch": two or three short things that would confirm the pessimistic case}'
        )
        data = self.chat(self.deep_model, prompt, max_tokens)

        points = []
        for entry in data.get("what_next") or []:
            if isinstance(entry, dict) and text(entry.get("point"), 10):
                points.append({
                    "point": text(entry["point"], 360),
                    "direction": choice(entry.get("direction"), DIRECTIONS, "mixed"),
                    "timeframe": choice(entry.get("timeframe"), TIMEFRAMES, "weeks"),
                    "confidence": choice(entry.get("confidence"), CONFIDENCE, "medium"),
                    "stories": id_list(entry.get("stories"), story_keys),
                })
        if not points:
            raise AIError("outlook reply had no what_next points")

        by_name = {}
        for entry in data.get("drivers") or []:
            if isinstance(entry, dict):
                by_name[str(entry.get("name", "")).strip().lower()] = entry
        driver_rows = []
        known_names = {d.strip().lower() for d in drivers}
        for index, name in enumerate(drivers):
            entry = by_name.get(name.strip().lower())
            if entry is None and index < len(data.get("drivers") or []):
                # Only borrow the entry in the same position when it does not belong to another driver.
                candidate = data["drivers"][index]
                if isinstance(candidate, dict) and str(candidate.get("name", "")).strip().lower() not in known_names:
                    entry = candidate
            entry = entry or {}
            driver_rows.append({
                "name": name,
                "status": choice(entry.get("status"), DRIVER_STATUS, "steady"),
                "reason": text(entry.get("reason"), 240),
                "stories": id_list(entry.get("stories"), story_keys),
            })

        def short_list(value):
            return [text(v, 160) for v in (value or []) if isinstance(v, str) and v.strip()][:3]

        return {
            "summary": text(data.get("summary"), 600),
            "what_next": points[:5],
            "drivers": driver_rows,
            "bull_watch": short_list(data.get("bull_watch")),
            "bear_watch": short_list(data.get("bear_watch")),
            "model": self.deep_model,
        }


# ---------------------------------------------------------------------------
# Quick model test: python scripts/ai.py MODEL_NAME
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = os.environ.get("HKBU_GENAI_API_KEY", "").strip()
    if not key:
        sys.exit("Set HKBU_GENAI_API_KEY first.")
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.8-flash"
    ai = AI(key, model, model)
    reply = ai.chat(model, 'Return {"ok": true, "model_says": "one short sentence about forward P/E"}', 1500)
    print(json.dumps(reply, indent=2))
    print("tokens used:", dict(ai.tokens), "| JSON mode off for:", sorted(ai.json_mode_off) or "none")
