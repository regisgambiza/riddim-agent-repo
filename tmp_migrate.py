import sqlite3
conn = sqlite3.connect('riddims_multi_year_new.db')
cur = conn.cursor()
cols = [r[1] for r in cur.execute('pragma table_info(riddims)')]
print('Current columns:', cols)
add_sqls = [
    'ALTER TABLE riddims ADD COLUMN spotify_id TEXT',
    'ALTER TABLE riddims ADD COLUMN spotify_url TEXT',
    'ALTER TABLE riddims ADD COLUMN producer TEXT',
    'ALTER TABLE riddims ADD COLUMN genre TEXT',
    'ALTER TABLE riddims ADD COLUMN source TEXT',
    'ALTER TABLE riddims ADD COLUMN source_url TEXT',
    'ALTER TABLE riddims ADD COLUMN match_status TEXT DEFAULT "UNVERIFIED"',
    'ALTER TABLE riddims ADD COLUMN notes TEXT',
]
for sql in add_sqls:
    col_name = sql.split()[4].replace('"', '')
    if col_name not in cols:
        try:
            cur.execute(sql)
            print(f'Added: {col_name}')
        except sqlite3.OperationalError as e:
            print(f'Already exists or error: {col_name}: {e}')
conn.commit()
conn.close()
print('Done')