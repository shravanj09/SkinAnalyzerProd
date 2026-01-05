import sys
sys.path.insert(0, 'services/api-gateway/app')
from utils.category_hierarchy import CATEGORY_HIERARCHY

print('Category Total Planned Verification:')
print('=' * 80)
total_all = 0
errors = []

for cat_name, cat_data in CATEGORY_HIERARCHY.items():
    total_planned = cat_data.get('total_planned', 0)

    # Count actual features
    actual_count = 0
    for model_config in cat_data.get('models', {}).values():
        actual_count += len(model_config.get('features', []))

    total_all += total_planned
    status = 'OK' if total_planned == actual_count else 'ERROR'

    if total_planned != actual_count:
        errors.append(f'{cat_name}: planned={total_planned}, actual={actual_count}')

    print(f'[{status}] {cat_name}: {total_planned} planned, {actual_count} actual')

print('=' * 80)
print(f'Total across all categories: {total_all}')
print(f'Target: 184 unique features')

if errors:
    print(f'\nERRORS FOUND ({len(errors)}):')
    for err in errors:
        print(f'   {err}')
else:
    print('\nAll counts are correct!')
