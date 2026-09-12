import sqlite3
db = sqlite3.connect(r'C:\Users\regis\Downloads\Compressed\riddim_agent_2\riddim_agent\riddims_multi_year_new.db')
db.row_factory = sqlite3.Row
for t in ['riddims','tracks','alt_names']:
    print('===', t, '===')
    for c in db.execute('PRAGMA table_info(' + t + ')'):
        print('  ', c['name'], '-', c['type'])
print('Total UNVERIFIED:', db.execute('SELECT COUNT(*) FROM riddims WHERE match_status="UNVERIFIED"').fetchone()[0])
print('Total tracks:', db.execute('SELECT COUNT(*) FROM tracks').fetchone()[0])
print('Total alt_names:', db.execute('SELECT COUNT(*) FROM alt_names').fetchone()[0])
db.close()