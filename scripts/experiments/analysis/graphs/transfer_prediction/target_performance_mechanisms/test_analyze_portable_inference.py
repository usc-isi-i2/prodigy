import copy
import unittest

from .analyze_portable_inference import TARGETS, SOURCES, expected_conditions, validate


def fixture():
    models = [f'memberctl_{s}_lowest_sorted_s{n}' for s in sorted(SOURCES) for n in range(3)]
    datasets = [f'{t}/{s}' for t in sorted(TARGETS) for s in ('original','fresh')]
    protocol={'partial_replay':False,'new_training':False,'model_ids':models,'dataset_ids':datasets,
              'conditions':list(expected_conditions()),'code_manifest_sha256':'fixture','environment':{'fixture':True}}
    done={'partial_replay':False,'cells':2700,'model_input_cells':60,'reference_conditions':2580,
          'all_inputs_weights_unchanged':True,'research_checkout_imported':False,'elapsed_seconds':1.}
    rows,receipts,audits=[],[],[]
    refs={'topology':[],'message':[],'dose':[]}
    counts={'covid_political':3072,'election2020':256,'facebook_page_reference':1024,'twibot20':3072,'ukr_rus_suspended':256}
    for dataset in datasets:
        target,stream=dataset.split('/')
        for source in sorted(SOURCES):
            for seed in range(3):
                mid=f'memberctl_{source}_lowest_sorted_s{seed}'
                common={'target':target,'stream':stream,'source':source,'seed':seed,'model_id':mid,
                        'step':2500,'roc_auc':.7,'accuracy':.6,'f1':.5,'nll':.8}
                for condition in expected_conditions():
                    rows.append({**common,'dataset_id':dataset,'condition':condition})
                    parts=condition.split('/')
                    if parts[0]=='topology': refs['topology'].append({**common,'query_condition':parts[1],'support_condition':parts[2],'draw':int(parts[3])})
                    if parts[0]=='message': refs['message'].append({**common,'condition':parts[1],'role':parts[2]})
                    if parts[0]=='dose': refs['dose'].append({**common,'suppression_percent':int(parts[1]),'draw':int(parts[2])})
                receipts.append({'dataset_id':dataset,'model_id':mid,'batches':32,'query_occurrences':counts[target],
                    'reference_conditions':43,'query_pre_exact':288,'query_post_exact':448,'direct_role_checks':37,
                    'input_and_model_unchanged':True,'operator_restored':True,'max_logit_error':0.,'max_metric_error':0.})
                audits.append({'target':target,'stream':stream,'model_id':mid,'constructor_and_tensor_adapter_bit_exact':True})
    exported={'datasets':10,'models':6,'batches':320,'constructor_checks':audits,'code_manifest_sha256':'fixture',
              'account_ids_removed':True,'features_remain_private':True}
    return protocol,done,rows,receipts,exported,refs


class PortableValidation(unittest.TestCase):
    def test_complete_grid(self):
        result=validate(*fixture())
        self.assertEqual(result['canonical_reference_cells'],2580)
        self.assertEqual(result['query_pre_exact_checks'],17280)
        self.assertEqual(result['query_post_exact_checks'],26880)
        self.assertEqual(result['direct_role_checks'],2220)

    def test_partial_and_missing_reference_rejected(self):
        args=fixture(); args[1]['partial_replay']=True
        with self.assertRaises(ValueError): validate(*args)
        args=fixture(); args[3][0]['reference_conditions']=42
        with self.assertRaises(ValueError): validate(*args)
        args=fixture(); args[5]['dose'].pop()
        with self.assertRaises(ValueError): validate(*args)

    def test_wrong_logit_metric_and_constructor_rejected(self):
        args=fixture(); args[3][0]['max_logit_error']=.001
        with self.assertRaises(ValueError): validate(*args)
        args=fixture(); args[5]['topology'][0]['roc_auc']=.1
        with self.assertRaises(ValueError): validate(*args)
        args=fixture(); args[4]['constructor_checks'][0]['constructor_and_tensor_adapter_bit_exact']=False
        with self.assertRaises(ValueError): validate(*args)

    def test_missing_duplicate_and_nonfinite_cell_rejected(self):
        args=fixture(); args[2].pop()
        with self.assertRaises(ValueError): validate(*args)
        args=fixture(); args[2][-1]=copy.deepcopy(args[2][0])
        with self.assertRaises(ValueError): validate(*args)
        args=fixture(); next(r for r in args[2] if r['condition']=='prototype/raw')['roc_auc']=float('nan')
        with self.assertRaises(ValueError): validate(*args)


if __name__=='__main__':unittest.main()
