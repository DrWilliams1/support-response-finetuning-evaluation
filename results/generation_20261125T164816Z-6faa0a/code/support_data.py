"""Original synthetic fictional-store cases and strict, limited output metrics."""
import collections
import hashlib
import json
import random

CATEGORIES = ('damaged_item', 'missing_receipt', 'outside_window', 'used_item',
              'eligible', 'eligible_boundary')
KEYS = {'ticket_id', 'decision', 'reason', 'reply'}
REPLIES = {
    'damaged_item': 'Your damaged item needs a support review before a return decision can be made.',
    'missing_receipt': 'Please provide your receipt so support can review your return request.',
    'outside_window': 'Your request is outside the stated return window, so this return cannot be approved.',
    'used_item': 'This return cannot be approved because the policy requires an unused item.',
    'eligible': 'Your unused item meets the receipt and return-window requirements, so the return is approved.',
}
POLICY = '''You draft responses for a fictional store. Use only the supplied facts.
Apply these rules in this exact order:
1. If damaged is true, decision=review and reason=damaged_item, regardless of other facts.
2. Otherwise, if receipt is false, decision=review and reason=missing_receipt.
3. Otherwise, if elapsed_days > return_window_days, decision=deny and reason=outside_window.
4. Otherwise, if unused is false, decision=deny and reason=used_item.
5. Otherwise decision=approve and reason=eligible. The deadline day is included.
Return one JSON object only with exactly four string fields: ticket_id, decision, reason, reply.
Copy ticket_id exactly. Write a helpful reply of 1 to 60 words consistent with the decision.
Do not add markdown, extra keys, promised payments, or unsupported facts.'''

def rule_decision(facts):
    """Explicit business-rule baseline; no learned model is needed for these facts."""
    if facts['damaged']: return 'review', 'damaged_item'
    if not facts['receipt']: return 'review', 'missing_receipt'
    if facts['elapsed_days'] > facts['return_window_days']: return 'deny', 'outside_window'
    if not facts['unused']: return 'deny', 'used_item'
    return 'approve', 'eligible'

def category(facts):
    _, reason = rule_decision(facts)
    if reason == 'eligible' and facts['elapsed_days'] == facts['return_window_days']:
        return 'eligible_boundary'
    return reason

def fact_key(facts):
    # IDs and prose cannot turn identical policy inputs into distinct split groups.
    return tuple(facts[k] for k in ('return_window_days','elapsed_days','receipt','unused','damaged'))

def rule_response(row):
    decision, reason = rule_decision(row['facts'])
    return json.dumps({'ticket_id':row['id'],'decision':decision,'reason':reason,
                       'reply':REPLIES[reason]}, ensure_ascii=False)

def build_dataset(seed=2026):
    rng=random.Random(seed)
    pools={c:[] for c in CATEGORIES}
    # Complete finite candidate pool; every fact tuple appears once.
    for window in range(7,181):
        for elapsed in range(0,201):
            for receipt in (False,True):
                for unused in (False,True):
                    for damaged in (False,True):
                        f={'return_window_days':window,'elapsed_days':elapsed,
                           'receipt':receipt,'unused':unused,'damaged':damaged}
                        pools[category(f)].append(f)
    out={s:[] for s in ('train','development','test')}
    leadins=('Please handle this return request.', 'Draft a response for this support ticket.',
             'Apply the store policy to this request.', 'Respond to the customer using these facts.')
    for cat in CATEGORIES:
        pool=pools[cat];rng.shuffle(pool)
        for split,start,count in [('train',0,80),('development',80,10),('test',90,20)]:
            for i,facts in enumerate(pool[start:start+count]):
                key=json.dumps(fact_key(facts),separators=(',',':'))
                ticket='T-'+hashlib.sha256(key.encode()).hexdigest()[:10].upper()
                row={'id':ticket,'facts':facts,'category':cat,'split':split}
                prompt_facts={'ticket_id':ticket,**facts}
                row['prompt']=POLICY+'\n\n'+leadins[rng.randrange(len(leadins))]+'\n'+json.dumps(prompt_facts)
                row['completion']=rule_response(row)
                bad=json.loads(row['completion']); kind=i%6
                if kind==0:
                    bad['decision']={'approve':'deny','deny':'review','review':'approve'}[bad['decision']]
                    error='wrong_decision'
                elif kind==1:
                    bad['reason']='eligible' if bad['reason']!='eligible' else 'used_item';error='wrong_reason'
                elif kind==2:
                    bad['ticket_id']='T-0000000000';error='wrong_ticket'
                elif kind==3:
                    bad['extra']='unrequested';error='extra_key'
                elif kind==4:
                    error='markdown_wrapper'
                else:
                    bad['reply']='A payment of one hundred dollars is guaranteed to arrive tomorrow.';error='unsupported_reply'
                rejected=json.dumps(bad)
                row['rejected']='```json\n'+rejected+'\n```' if kind==4 else rejected
                row['rejection_type']=error
                out[split].append(row)
    for rows in out.values(): rng.shuffle(rows)
    validate_dataset(out)
    return out

def validate_dataset(data):
    all_ids=set(); all_facts=set();all_prompts=set()
    for split,expected in [('train',80),('development',10),('test',20)]:
        rows=data[split]
        if collections.Counter(r['category'] for r in rows)!=collections.Counter({c:expected for c in CATEGORIES}):
            raise ValueError('Unexpected split balance')
        for r in rows:
            if r['split']!=split or r['category']!=category(r['facts']): raise ValueError('Split/category mismatch')
            key=fact_key(r['facts'])
            if r['id'] in all_ids or key in all_facts or r['prompt'] in all_prompts: raise ValueError('Repeated ID, facts, or prompt')
            all_ids.add(r['id']);all_facts.add(key);all_prompts.add(r['prompt'])
            if r['completion']!=rule_response(r): raise ValueError('Reference violates policy')
            if r['rejected']==r['completion']: raise ValueError('Preference pair is identical')
    return {'rows':{s:len(r) for s,r in data.items()},'unique_fact_groups':len(all_facts),
            'shared_fact_groups':0,'synthetic':True,'wording_policy':'Shared fixed templates; new fact combinations, not unseen language.'}

def _strict_object(pairs):
    obj={}
    for key,value in pairs:
        if key in obj: raise ValueError('Duplicate JSON key')
        obj[key]=value
    return obj

def score_response(row,text):
    score={k:False for k in ('json_valid','schema_valid','decision_correct','reason_correct',
                             'ticket_correct','structured_pass','reference_reply_exact')}
    try:
        value=json.loads(text,object_pairs_hook=_strict_object,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonstandard JSON constant')))
        score['json_valid']=True
    except (ValueError,TypeError): return score
    if not isinstance(value,dict): return score
    score['schema_valid']=(set(value)==KEYS and all(isinstance(v,str) for v in value.values())
        and 1<=len(value.get('reply','').split())<=60 and value.get('decision') in ('approve','deny','review')
        and value.get('reason') in REPLIES)
    truth=json.loads(row['completion'])
    for field,metric in [('decision','decision_correct'),('reason','reason_correct'),('ticket_id','ticket_correct')]:
        score[metric]=value.get(field)==truth[field]
    score['structured_pass']=all(score[k] for k in ('schema_valid','decision_correct','reason_correct','ticket_correct'))
    score['reference_reply_exact']=value.get('reply')==truth['reply']
    return score

def summarize(rows,outputs):
    if len(rows)!=len(outputs) or not rows: raise ValueError('Expected aligned nonempty outputs')
    scores=[score_response(r,t) for r,t in zip(rows,outputs)]
    result={'n_examples':len(rows)}
    for name in scores[0]: result[name]=sum(s[name] for s in scores)/len(scores)
    result['per_category']={c:{'n_examples':sum(r['category']==c for r in rows),
        'structured_pass':sum(s['structured_pass'] for r,s in zip(rows,scores) if r['category']==c)/sum(r['category']==c for r in rows)}
        for c in CATEGORIES if any(r['category']==c for r in rows)}
    return result
