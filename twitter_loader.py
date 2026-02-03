"""
Test script for loading Twitter CSV data
"""
import sys
sys.path.insert(0, '/Users/amandeep/Github/prodigy')

from data.twitter_csv import get_twitter_dataset

root = '/Users/amandeep/Github/prodigy/datasets/twitter'
csv_file = 'midterm-2022-10-01-21.csv'

print('Testing Twitter CSV loader...')
print(f'Root: {root}')
print(f'CSV file: {csv_file}')
print()

print('=== Loading with numerical features ===')
dataset = get_twitter_dataset(
    root=root,
    csv_filename=csv_file,
    label_type='verified',
    n_hop=2,
    original_features=True,
    max_users=1000
)

print(f'Dataset created: {len(dataset)} nodes')
print(f'Graph info: {dataset.graph}')
print('Test complete!')
