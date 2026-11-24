-- Read-only role for docqa-agent. Before running: replace 'change-me' with a long
-- random password and neondb with your database name.
CREATE ROLE docqa_agent_ro WITH LOGIN PASSWORD 'change-me';
GRANT CONNECT ON DATABASE neondb TO docqa_agent_ro;
GRANT USAGE ON SCHEMA public TO docqa_agent_ro;

-- Only the two tables the server reads. "Account" holds users' OAuth tokens and stays off limits.
GRANT SELECT ON "Document", "Chunk" TO docqa_agent_ro;
