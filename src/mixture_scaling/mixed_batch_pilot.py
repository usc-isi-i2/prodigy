"""Three diagnostic pairs; balanced source losses before each AdamW step."""
import sys
from . import interleaved_mlp_pairs as m

PAIRS=(('ukr_rus_twitter','facebook_page_reference'),
       ('covid19_twitter','midterm'),
       ('covid_political','ukr_rus_suspended'))

def pair_rows():
    return [(f'mixed_{a}__and__{b}',a,b) for a,b in PAIRS]

def main():
    m.pair_rows=pair_rows
    sys.argv.extend(['--batch-schedule','mixed'])
    m.main()

if __name__=='__main__':main()
