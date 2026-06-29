import difflib
import re

from neo4j import GraphDatabase, NotificationMinimumSeverity

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class KnowledgeGraphManager:
    _driver = None
    _constraints_created = False

    def __init__(self):
        if KnowledgeGraphManager._driver is None:
            KnowledgeGraphManager._driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                notifications_min_severity=NotificationMinimumSeverity.OFF,
            )
        self.driver = KnowledgeGraphManager._driver
        if not KnowledgeGraphManager._constraints_created:
            self._ensure_constraints()
            KnowledgeGraphManager._constraints_created = True

    def _ensure_constraints(self):
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT post_title IF NOT EXISTS "
                "FOR (p:Post) REQUIRE p.title IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT topic_name IF NOT EXISTS "
                "FOR (t:Topic) REQUIRE t.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT claim_text IF NOT EXISTS "
                "FOR (c:Claim) REQUIRE c.text IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT source_url IF NOT EXISTS "
                "FOR (s:Source) REQUIRE s.url IS UNIQUE"
            )

    # ------------------------------------------------------------------ #
    # PLANNING                                                             #
    # ------------------------------------------------------------------ #

    def get_recent_posts(self, n: int = 5) -> list[dict]:
        with self.driver.session() as session:
            return session.run(
                "MATCH (p:Post) "
                "WHERE p.planned_date IS NOT NULL AND p.planned_date <> '' "
                "RETURN p.title AS title, p.topic AS topic, "
                "p.post_type AS post_type, p.planned_date AS planned_date "
                "ORDER BY p.planned_date DESC LIMIT $n",
                n=n
            ).data()

    def get_last_scheduled_date(self) -> str | None:
        """Restituisce la data pianificata più recente tra tutti i post nel KG."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Post) "
                "WHERE p.planned_date IS NOT NULL AND p.planned_date <> '' "
                "RETURN p.planned_date AS d ORDER BY p.planned_date DESC LIMIT 1"
            ).single()
        return result["d"] if result else None

    def get_posts_scheduled_for_date(self, date_str: str) -> list[dict]:
        """Restituisce i post con planned_date uguale alla data fornita (ISO YYYY-MM-DD)."""
        with self.driver.session() as session:
            return session.run(
                "MATCH (p:Post) WHERE p.planned_date = $date "
                "RETURN p.title AS title, p.topic AS topic, p.post_type AS post_type",
                date=date_str
            ).data()

    def get_coverage_gaps(self, domain_topics: list[str]) -> list[str]:
        with self.driver.session() as session:
            covered = {
                r["name"].lower()
                for r in session.run(
                    "MATCH (:Post)-[:TRATTA]->(t:Topic) RETURN DISTINCT t.name AS name"
                ).data()
            }
        return [t for t in domain_topics if t.lower() not in covered]

    def get_uncovered_related_topics(self) -> list[str]:
        with self.driver.session() as session:
            covered = {
                r["name"].lower()
                for r in session.run(
                    "MATCH (:Post)-[:TRATTA]->(t:Topic) RETURN DISTINCT t.name AS name"
                ).data()
            }
            related = session.run(
                "MATCH (:Topic)-[:CORRELATO_A]->(t:Topic) RETURN DISTINCT t.name AS name"
            ).value("name")
        seen = set()
        uncovered = []
        for name in related:
            if name and name.lower() not in covered and name.lower() not in seen:
                uncovered.append(name)
                seen.add(name.lower())
        return uncovered

    # ------------------------------------------------------------------ #
    # DRAFTING                                                             #
    # ------------------------------------------------------------------ #

    def get_topic_context(self, current_topic: str) -> str:
        with self.driver.session() as session:
            related_posts = session.run(
                "MATCH (p:Post)-[:TRATTA]->(t:Topic) "
                "WHERE toLower(t.name) CONTAINS toLower($topic) "
                "   OR toLower(p.topic) CONTAINS toLower($topic) "
                "RETURN p.title AS title, p.post_type AS post_type",
                topic=current_topic
            ).data()

            if not related_posts:
                return (
                    f"'{current_topic}' è un topic nuovo per il blog. "
                    "Gap di copertura identificato: nessun vincolo di coerenza."
                )

            claims = session.run(
                "MATCH (p:Post)-[:CONTIENE_CLAIM]->(c:Claim) "
                "WHERE p.title IN $titles RETURN c.text AS claim",
                titles=[p["title"] for p in related_posts]
            ).value("claim")

        parts = ["Post correlati già pubblicati:"]
        for p in related_posts:
            parts.append(f"  - '{p['title']}' ({p.get('post_type', 'N/A')})")
        if claims:
            parts.append("\nClaim già sostenuti (mantieni coerenza):")
            for c in claims:
                parts.append(f"  - {c}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # K-RAG                                                                #
    # ------------------------------------------------------------------ #

    def expand_query_for_rag(self, topic: str) -> str:
        with self.driver.session() as session:
            related = session.run(
                "MATCH (t:Topic)-[:CORRELATO_A]-(r:Topic) "
                "WHERE toLower(t.name) = toLower($topic) "
                "RETURN r.name AS name LIMIT 3",
                topic=topic
            ).value("name")

            past_claims = session.run(
                "MATCH (p:Post)-[:TRATTA]->(t:Topic), (p)-[:CONTIENE_CLAIM]->(c:Claim) "
                "WHERE toLower(t.name) = toLower($topic) "
                "RETURN c.text AS claim LIMIT 2",
                topic=topic
            ).value("claim")

        parts = [topic] + list(related) + list(past_claims)
        return ", ".join(p for p in parts if p)


    # ------------------------------------------------------------------ #
    # AGGIORNAMENTO INCREMENTALE                                           #
    # ------------------------------------------------------------------ #

    def add_approved_post(
        self,
        title: str,
        topic: str,
        post_type: str,
        claims: list[str],
        sources: list[str],
        related_topics: list[str] | None = None,
        planned_date: str | None = None,
    ):
        with self.driver.session() as session:
            session.run(
                "MERGE (p:Post {title: $title}) "
                "SET p.topic = $topic, p.post_type = $post_type, "
                "p.planned_date = $planned_date",
                title=title, topic=topic, post_type=post_type,
                planned_date=planned_date or "",
            )
            session.run("MERGE (t:Topic {name: $name})", name=topic)
            session.run(
                "MATCH (p:Post {title: $title}), (t:Topic {name: $topic}) "
                "MERGE (p)-[:TRATTA]->(t)",
                title=title, topic=topic
            )

            for claim in claims:
                session.run("MERGE (c:Claim {text: $text})", text=claim)
                session.run(
                    "MATCH (p:Post {title: $title}), (c:Claim {text: $claim}) "
                    "MERGE (p)-[:CONTIENE_CLAIM]->(c)",
                    title=title, claim=claim
                )

            for source in sources:
                session.run("MERGE (s:Source {url: $url})", url=source)
                session.run(
                    "MATCH (p:Post {title: $title}), (s:Source {url: $source}) "
                    "MERGE (p)-[:USA_FONTE]->(s)",
                    title=title, source=source
                )

            if related_topics:
                existing = session.run(
                    "MATCH (t:Topic) RETURN t.name AS name"
                ).value("name")
                for rel_topic in related_topics:
                    normalized = self._normalize_topic(existing, rel_topic)
                    session.run("MERGE (t:Topic {name: $name})", name=normalized)
                    session.run(
                        "MATCH (t1:Topic {name: $topic}), (t2:Topic {name: $rel}) "
                        "MERGE (t1)-[:CORRELATO_A]->(t2)",
                        topic=topic, rel=normalized
                    )

        print(f"[KG] Aggiornato: '{title}' ({post_type}) — "
              f"{len(claims)} claim, {len(sources)} fonti")

    # ------------------------------------------------------------------ #
    # UTILITY PRIVATA                                                      #
    # ------------------------------------------------------------------ #

    def _normalize_topic(self, existing: list[str], name: str) -> str:
        if not existing:
            return name
        if name in existing:
            return name
        stripped = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
        if stripped != name:
            for e in existing:
                if e.lower() == stripped.lower():
                    return e
        matches = difflib.get_close_matches(name, existing, n=1, cutoff=0.88)
        return matches[0] if matches else name
