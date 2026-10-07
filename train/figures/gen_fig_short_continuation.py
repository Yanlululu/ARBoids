"""The second and final confirmation uses 97.5% confidence intervals."""
import json

import gen_fig_predictive_interception as analysis
from evaluate_short_continuation import ROOT,METHODS,SELECTED,frozen_inputs


def main():
    spec=json.loads((ROOT/'specification.json').read_text(encoding='utf-8'))
    if spec!=json.loads(json.dumps(frozen_inputs())):
        raise AssertionError('Fixed follow-up inputs changed.')
    if not json.loads((ROOT/'confirm_summary.json').read_text(encoding='utf-8'))['passed']:
        raise AssertionError('Follow-up did not pass execution checks.')
    analysis.METHODS=METHODS
    analysis.CONFIDENCE=.975
    analysis.LABELS.update(long_value='20-s continuation',predictive='8-s continuation')
    analysis.COLORS['long_value']='#56B4E9'
    rows=analysis.read_csv(ROOT/'confirm.csv')
    table=analysis.summarize(rows)
    contrasts=analysis.comparisons(rows)
    analysis.save_csv(ROOT/'outcome_summary.csv',table)
    analysis.save_csv(ROOT/'paired_comparisons.csv',contrasts)
    analysis.save_csv(ROOT/'cell_summary.csv',analysis.summarize_cells(rows))
    analysis.figures(rows,table,contrasts,ROOT)
    primary=[r for r in contrasts if r['group']==6 and r['reference'] in ('cbf','best_fixed')]
    strong=all(r['improvement_percent']>=10 and r['delta_ci_high']<0 and r['delta_success_pp']>=0
               and r['collisions_new']<=r['collisions_reference'] for r in primary)
    # The unchanged 20-second method was a declared comparator. These are
    # secondary contrasts, never a replacement for the 8-second primary test.
    secondary_methods=('cbf','long_value','short_value','best_fixed')
    secondary_rows=[dict(r,method='predictive' if r['method']=='long_value' else r['method'])
                    for r in rows if r['method'] in secondary_methods]
    analysis.METHODS=('cbf','predictive','short_value','best_fixed')
    secondary=[r for r in analysis.comparisons(secondary_rows) if r['group']==6]
    output=dict(passed=True,confidence_level=.975,primary_strong_advantage=strong,primary=primary,
                selected=SELECTED,long_value_secondary=secondary)
    (ROOT/'analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output))


if __name__=='__main__':
    main()
