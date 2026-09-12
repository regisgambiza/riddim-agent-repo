import sys
sys.path.insert(0, '.')
import db
import matching

records = db.list_all_riddims()
print(f"Total records: {len(records)}")
ranked = matching.rank_candidates("Bombay Riddim 2022", records, top_n=10)
print("\n--- Top 10 for 'Bombay Riddim 2022' ---")
for c in ranked[:10]:
    print(f"  id={c['candidate_id']} name={c['name']!r} year={c['year']} scores={c['scores']}")
print("\n--- Checking normalize ---")
print("normalize('Bombay Riddim 2022'):", db.normalize("Bombay Riddim 2022"))
print("normalize('Bombay Riddim (2022)'):", db.normalize("Bombay Riddim (2022)"))
# Check Bombay records in DB
bombay_records = [r for r in records if 'Bombay' in r.get('riddim_name','').lower()]
print(f"\nTotal Bombay records: {len(bombay_records)}")
for r in bombay_records[:10]:
    print(f"  id={r['id']} name={r['riddim_name']!r} year={r.get('year')}")