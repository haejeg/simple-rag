'''
    Eval dashboard.

    Runs every test question through POST /query_similarity and logs the run together with
    the backend's hyperparameters (GET /config), so you can see how each change moved the score.
    Hover a point on the chart to see the hyperparameters that run used.

    Files (project root):
        evals/questions.txt   one test question per line (editable from this page)
        evals/runs.jsonl      one JSON line per run, appended after each run
'''
import json
import os
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv('RAG_API_URL', 'http://127.0.0.1:8000')
ROOT = Path(__file__).resolve().parents[2]
EVAL_DIRECTORY = ROOT / 'evals'
QUESTIONS_FILE = EVAL_DIRECTORY / 'questions.txt'
RUNS_FILE = EVAL_DIRECTORY / 'runs.jsonl'
METRIC = 'query_similarity' # Mean of the best answer-to-chunk cosine similarity per question
QUERY_TIMEOUT = 300

ACCENT = '#4c5fd5'
ACCENT_SOFT = '#b9c0ef'

st.set_page_config(page_title='Evals', page_icon=':material/monitoring:', layout='wide')

st.html('''
<style>
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }
h1 { font-weight: 500 !important; letter-spacing: -0.01em; }
.muted { color: #6b6862; font-size: 0.85rem; margin: 0; }
</style>''')


# ---- Helpers ----

def api_call(method, path, **kwargs):
    '''
        Calls the backend and returns (json, error_message).
    '''
    try:
        response = requests.request(method, f'{API_URL}{path}', **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.ConnectionError:
        return None, f"Can't reach the backend at {API_URL}. Is uvicorn main:app running?"
    except requests.Timeout:
        return None, 'The backend took too long to respond.'
    except requests.RequestException as e:
        return None, f'Backend error: {e}'

def load_questions():
    if not QUESTIONS_FILE.is_file():
        return []
    return [line.strip() for line in QUESTIONS_FILE.read_text(encoding='utf-8').splitlines() if line.strip()]

def save_questions(questions):
    EVAL_DIRECTORY.mkdir(exist_ok=True)
    QUESTIONS_FILE.write_text('\n'.join(questions) + '\n', encoding='utf-8')

def load_runs():
    if not RUNS_FILE.is_file():
        return []
    runs = []
    for line in RUNS_FILE.read_text(encoding='utf-8').splitlines():
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            pass # Skip a half-written or hand-edited line instead of breaking the page
    return runs

def save_runs(runs):
    EVAL_DIRECTORY.mkdir(exist_ok=True)
    RUNS_FILE.write_text(''.join(json.dumps(r) + '\n' for r in runs), encoding='utf-8')

def append_run(run):
    EVAL_DIRECTORY.mkdir(exist_ok=True)
    with RUNS_FILE.open('a', encoding='utf-8') as f:
        f.write(json.dumps(run) + '\n')

def changes(previous, current):
    '''
        Describes which hyperparameters differ from the previous run, e.g. "TOP_N 4 → 5".
    '''
    if previous is None:
        return 'first run'
    diff = [f'{k} {previous.get(k, "—")} → {v}' for k, v in current.items() if previous.get(k) != v]
    return ', '.join(diff) or 'no change'

def run_eval(questions, note):
    '''
        Asks every question, then saves the run. Returns an error message, or None on success.
    '''
    config, error = api_call('GET', '/config', timeout=10)
    if error:
        if '404' in error:
            return 'The backend has no /config endpoint yet. Restart uvicorn so it picks up the latest main.py.'
        return error

    results = []
    progress = st.progress(0.0)
    for i, question in enumerate(questions):
        progress.progress(i / len(questions), text=f'Question {i + 1} of {len(questions)}: {question}')
        start = time.perf_counter()
        result, error = api_call('POST', '/query_similarity', json={'user_query': question}, timeout=QUERY_TIMEOUT)
        seconds = round(time.perf_counter() - start, 1)
        if error:
            results.append({'question': question, 'error': error, 'seconds': seconds})
        else:
            results.append({'question': question, 'similarity': result['similarity'],
                            'chunk_scores': result['chunk_scores'], 'sources': result['sources'],
                            'response': result['response'], 'seconds': seconds})
    progress.empty()

    scores = [r['similarity'] for r in results if 'similarity' in r]
    if not scores:
        return f'Every question failed, so the run was not saved. First error: {results[0]["error"]}'

    runs = load_runs()
    append_run({
        'id': max((r['id'] for r in runs), default=0) + 1,
        'time': datetime.now().isoformat(timespec='seconds'),
        'note': note.strip(),
        'metric': METRIC,
        'config': config,
        'mean': sum(scores) / len(scores),
        'min': min(scores),
        'max': max(scores),
        'failed': len(results) - len(scores),
        'results': results,
    })
    return None


# ---- Header and run controls ----

runs = load_runs()

st.title('Evals')
st.html('<p class="muted">Each run asks every test question through <code>/query_similarity</code> and logs '
        'the score with the backend\'s hyperparameters. Change a value in main.py, restart uvicorn '
        '(and rebuild the index if you changed chunking or the embedding model), then run again.</p>')

with st.container(border=True):
    questions = load_questions()
    with st.expander(f'Test questions ({len(questions)})'):
        edited = st.text_area('One question per line', '\n'.join(questions), height=220,
                              label_visibility='collapsed')
        if st.button('Save questions', icon=':material/save:'):
            save_questions([q.strip() for q in edited.splitlines() if q.strip()])
            st.rerun()

    with st.container(horizontal=True, vertical_alignment='bottom'):
        note = st.text_input('Note', placeholder='What did you change? (optional)', label_visibility='collapsed')
        clicked = st.button('Run test', type='primary', icon=':material/play_arrow:', disabled=not questions)

    if clicked:
        st.caption('Stay on this page until the run finishes; leaving it stops the run.')
        error = run_eval(questions, note)
        if error:
            st.error(error)
        else:
            st.toast('Run saved', icon=':material/check_circle:')
            st.rerun()

if not runs:
    st.info('No runs yet. Press **Run test** to record the first one.', icon=':material/insights:')
    st.stop()


# ---- Headline numbers ----

rows = []
for i, run in enumerate(runs):
    previous = runs[i - 1]['config'] if i else None
    rows.append({
        'run': run['id'],
        'time': datetime.fromisoformat(run['time']).strftime('%b %d, %H:%M'),
        'note': run.get('note') or '—',
        'changed': changes(previous, run['config']),
        'mean': run['mean'], 'min': run['min'], 'max': run['max'],
        'questions': len(run['results']), 'failed': run.get('failed', 0),
        **run['config'],
    })
df = pd.DataFrame(rows)
config_keys = list(dict.fromkeys(k for run in runs for k in run['config']))

latest = runs[-1]
best = max(runs, key=lambda r: r['mean'])
left, middle, right = st.columns(3)
left.metric('Latest mean similarity', f"{latest['mean']:.3f}",
            delta=f"{latest['mean'] - runs[-2]['mean']:+.3f}" if len(runs) > 1 else None,
            help='Average over the test questions of the best answer-to-chunk cosine similarity')
middle.metric('Best run', f"#{best['id']}", help=f"Mean {best['mean']:.3f}")
right.metric('Runs logged', len(runs))


# ---- Chart: mean per run, with the spread of question scores ----

tooltip = (
    [alt.Tooltip('run:O', title='Run'), alt.Tooltip('time:N', title='Time'),
     alt.Tooltip('note:N', title='Note'), alt.Tooltip('changed:N', title='Changed'),
     alt.Tooltip('mean:Q', title='Mean', format='.3f'), alt.Tooltip('min:Q', title='Lowest question', format='.3f'),
     alt.Tooltip('max:Q', title='Highest question', format='.3f')]
    + [alt.Tooltip(f'{k}:N', title=k) for k in config_keys]
)

base = alt.Chart(df).encode(x=alt.X('run:O', title='Run', axis=alt.Axis(labelAngle=0)))
spread = base.mark_rule(strokeWidth=2, color=ACCENT_SOFT, strokeCap='round').encode(
    y=alt.Y('min:Q', title='Similarity', scale=alt.Scale(zero=False, nice=True)), y2='max:Q')
line = base.mark_line(strokeWidth=2, color=ACCENT).encode(y='mean:Q')
points = base.mark_point(filled=True, size=90, color=ACCENT, stroke='white', strokeWidth=2, opacity=1).encode(
    y='mean:Q')
hit_area = base.mark_point(size=900, opacity=0).encode(y='mean:Q', tooltip=tooltip) # Bigger than the dot, easier to hover

chart = (spread + line + points + hit_area).properties(height=320).configure_axis(
    grid=True, gridColor='#ece9e3', domain=False, tickColor='#e4e1db', labelColor='#6b6862', titleColor='#6b6862',
).configure_view(strokeWidth=0)

with st.container(border=True):
    st.markdown('**Mean similarity per run**')
    st.html('<p class="muted">Dot = mean across questions. Vertical bar = lowest to highest question. '
            'Hover a dot for the hyperparameters.</p>')
    st.altair_chart(chart, width='stretch')


# ---- Tables ----

tab_runs, tab_detail = st.tabs(['All runs', 'Run details'])

with tab_runs:
    st.dataframe(
        df[['run', 'time', 'note', 'mean', 'min', 'max', 'changed'] + config_keys].iloc[::-1],
        hide_index=True, width='stretch',
        column_config={
            'run': st.column_config.NumberColumn('Run', format='#%d'),
            'mean': st.column_config.NumberColumn('Mean', format='%.3f'),
            'min': st.column_config.NumberColumn('Min', format='%.3f'),
            'max': st.column_config.NumberColumn('Max', format='%.3f'),
        },
    )

with tab_detail:
    ids = [r['id'] for r in runs][::-1]
    chosen = st.selectbox('Run', ids, format_func=lambda i: f'Run #{i}', label_visibility='collapsed')
    run = next(r for r in runs if r['id'] == chosen)

    detail = pd.DataFrame([{
        'question': r['question'],
        'similarity': r.get('similarity'),
        'sources': ', '.join(r.get('sources', [])) or r.get('error', ''),
        'seconds': r['seconds'],
        'answer': r.get('response', ''),
    } for r in run['results']])
    st.dataframe(detail, hide_index=True, width='stretch', column_config={
        'similarity': st.column_config.ProgressColumn('Similarity', format='%.3f', min_value=0.0, max_value=1.0),
        'answer': st.column_config.TextColumn('Answer', width='large'),
    })

    with st.container(horizontal=True):
        with st.popover('Hyperparameters', icon=':material/tune:'):
            st.json(run['config'])
        if st.button('Delete run', icon=':material/delete:'):
            save_runs([r for r in runs if r['id'] != chosen])
            st.rerun()
