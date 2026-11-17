import os, sqlite3
DB = os.getenv("CLINADAPT_DB", "clinadapt.db")
def conn():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.executescript("""create table if not exists cases(id text primary key, created real, output text, model_version text, source text);
    create table if not exists audit(id integer primary key autoincrement, case_id text, action text, mode text, note text, ts real);""")
    if "source" not in {r["name"] for r in c.execute("pragma table_info(cases)")}:
        c.execute("alter table cases add column source text")
    return c
