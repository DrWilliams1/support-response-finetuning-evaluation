import json,unittest
from support_data import build_dataset,rule_decision,score_response,validate_dataset

class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data=build_dataset()
    def test_boundary_and_precedence(self):
        f=dict(return_window_days=30,elapsed_days=30,receipt=True,unused=True,damaged=False)
        self.assertEqual(rule_decision(f),('approve','eligible'))
        self.assertEqual(rule_decision(dict(f,elapsed_days=31)),('deny','outside_window'))
        self.assertEqual(rule_decision(dict(f,unused=False)),('deny','used_item'))
        self.assertEqual(rule_decision(dict(f,elapsed_days=31,receipt=False)),('review','missing_receipt'))
        self.assertEqual(rule_decision(dict(f,elapsed_days=31,receipt=False,damaged=True)),('review','damaged_item'))
    def test_groups_and_balances(self):
        self.assertEqual(validate_dataset(self.data)['unique_fact_groups'],660)
        self.assertEqual([len(x) for x in self.data.values()],[480,60,120])
    def test_truth_passes_but_not_every_rejected_fails_fields(self):
        for row in self.data['test']:
            self.assertTrue(score_response(row,row['completion'])['structured_pass'])
            if row['rejection_type']=='unsupported_reply':
                self.assertTrue(score_response(row,row['rejected'])['structured_pass'])
                self.assertFalse(score_response(row,row['rejected'])['reference_reply_exact'])
    def test_strict_json(self):
        r=self.data['test'][0]
        for text in [r['completion']+' prose','```'+r['completion']+'```','{"reply":NaN}',
                     '{"decision":"approve","decision":"deny"}']:
            self.assertFalse(score_response(r,text)['json_valid'])
        for value in [[],{'ticket_id':r['id']},dict(json.loads(r['completion']),extra=True),
                      dict(json.loads(r['completion']),reply=17),dict(json.loads(r['completion']),reply='word '*61)]:
            self.assertFalse(score_response(r,json.dumps(value))['schema_valid'])
    def test_duplicate_fact_group_rejected(self):
        import copy
        d=copy.deepcopy(self.data);d['test'][0]['facts']=d['train'][0]['facts']
        with self.assertRaises(ValueError): validate_dataset(d)

if __name__=='__main__': unittest.main()
