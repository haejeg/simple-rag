'''
    Chat interface for the simple RAG backend.

    This file never imports main.py. It talks to the FastAPI server over HTTP:
        POST   /query   -> ask a question
        POST   /ingest  -> rebuild the db from the data folder
        DELETE /db      -> clear the db
    and reads/writes the data folder directly to list, preview and upload sources.

    Run (from the project root, so .streamlit/config.toml is picked up):
        uvicorn main:app
        streamlit run frontend/app.py
'''
import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv('RAG_API_URL', 'http://127.0.0.1:8000')
ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = ROOT / os.getenv('RAG_DATA_DIR', 'sample-data') # Must match DATA_DIRECTORY in main.py
QUERY_TIMEOUT = 300 # Seconds; local models can be slow on the first call
INGEST_TIMEOUT = 600

st.set_page_config(page_title='Notebook', page_icon=':material/chat_bubble:', layout='centered')


# ---- Styling ----

st.html('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap');

:root {
    --ink: #1f1e1c;
    --ink-soft: #6b6862;
    --line: #e4e1db;
    --surface: #ffffff;
    --accent: #4c5fd5;
}

[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; padding-bottom: 7rem; }

@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }

/* Empty state */
.greeting { text-align: center; margin-top: 18vh; animation: rise 0.5s ease-out both; }
.greeting h1 {
    font-family: 'Source Serif 4', Georgia, serif !important; font-weight: 400 !important;
    font-size: 2.3rem !important; letter-spacing: -0.01em; padding: 0 0 0.4rem !important;
}
.greeting p { color: var(--ink-soft); font-size: 0.95rem; margin: 0; }

/* Composer: the chat input, both centred (empty state) and pinned at the bottom */
[data-testid="stChatInput"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: 22px;
    box-shadow: 0 2px 10px rgba(31, 30, 28, 0.05);
    transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.25s ease;
    animation: rise 0.5s ease-out 0.08s both;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(76, 95, 213, 0.45);
    box-shadow: 0 6px 28px rgba(76, 95, 213, 0.14), 0 0 0 4px rgba(76, 95, 213, 0.07);
    transform: translateY(-1px);
}
[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; border-radius: inherit; }
[data-testid="stChatInput"] textarea { font-size: 1rem; }
[data-testid="stChatInput"] button {
    border-radius: 12px; background: var(--accent); color: #fff; transition: transform 0.15s ease, opacity 0.15s ease;
}
[data-testid="stChatInput"] button:hover { transform: scale(1.06); }
[data-testid="stChatInput"] button:disabled { background: #d9d6cf; }
.st-key-composer_hero { margin-top: 1.6rem; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent; }
[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(to top, #faf9f7 70%, rgba(250, 249, 247, 0)); padding-top: 1.5rem;
}

/* Suggestions */
.st-key-suggestions { animation: rise 0.5s ease-out 0.16s both; margin-top: 0.4rem; }
.st-key-suggestions button {
    background: transparent; border: 1px solid var(--line); border-radius: 999px;
    color: var(--ink-soft); font-size: 0.85rem; padding: 0.3rem 0.9rem; min-height: 0;
    transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}
.st-key-suggestions button:hover { background: var(--surface); color: var(--ink); border-color: #cfcbc3; }
.st-key-suggestions button p { font-size: 0.85rem; }

/* Messages */
[class*="st-key-msg_"] { animation: rise 0.35s ease-out both; }
[class*="st-key-msg_user_"] {
    width: fit-content !important; max-width: 80%; margin-left: auto;
    background: #efece6; border-radius: 18px; padding: 0.6rem 1rem;
}
[class*="st-key-msg_user_"] p { margin: 0; }
[class*="st-key-msg_user_"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[class*="st-key-msg_bot_"] { padding: 0.2rem 0.1rem 0.6rem; line-height: 1.65; }
[class*="st-key-msg_bot_"] p, [class*="st-key-msg_bot_"] li { font-size: 1rem; }
.msg-error { color: #b4442b; font-size: 0.92rem; }

/* Thinking indicator */
.thinking {
    display: inline-block; font-size: 0.95rem; font-weight: 500;
    background: linear-gradient(90deg, #9b978f 0%, #9b978f 40%, #1f1e1c 50%, #9b978f 60%, #9b978f 100%);
    background-size: 200% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: shimmer 2.2s linear infinite;
}

/* Citation chips and message actions */
[class*="st-key-chip_"] button, [class*="st-key-save_"] button {
    min-height: 0; padding: 0.15rem 0.6rem; border-radius: 8px; font-size: 0.78rem;
    background: transparent; border: 1px solid var(--line); color: var(--ink-soft);
    transition: background 0.2s ease, color 0.2s ease;
}
[class*="st-key-chip_"] button:hover, [class*="st-key-save_"] button:hover { background: var(--surface); color: var(--ink); }
[class*="st-key-chip_"] button p, [class*="st-key-save_"] button p { font-size: 0.78rem; }

/* Sidebar */
.side-brand { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem; }
.side-brand b { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 500; }
.status { font-size: 0.75rem; color: var(--ink-soft); display: inline-flex; align-items: center; gap: 0.35rem; }
.status .dot { width: 7px; height: 7px; border-radius: 50%; }
.status.on .dot { background: #2f9e6b; }
.status.off .dot { background: #c2543b; }
.muted { color: var(--ink-soft); font-size: 0.82rem; line-height: 1.45; }
[class*="st-key-src_"] button {
    justify-content: flex-start; border: none; background: transparent; padding: 0.35rem 0.5rem; min-height: 0;
}
[class*="st-key-src_"] button:hover { background: rgba(31, 30, 28, 0.05); }
[class*="st-key-src_"] button > div { width: 100%; justify-content: flex-start; }
[class*="st-key-src_"] button [data-testid="stMarkdownContainer"] { text-align: left; overflow: hidden; }
[class*="st-key-src_"] button p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.88rem; }
[class*="st-key-note_"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 0.7rem 0.8rem;
    animation: rise 0.3s ease-out both;
}
[class*="st-key-note_"] p, [class*="st-key-note_"] li { font-size: 0.84rem; }
.note-title { font-weight: 600; font-size: 0.86rem; }
.note-sources { color: var(--ink-soft); font-size: 0.72rem; margin-top: 0.3rem; }
</style>
''')


# ---- State ----

state = st.session_state
state.setdefault('messages', []) # [{'role': 'user'|'assistant', 'content': str, 'sources': [str], 'error': bool}]
state.setdefault('notes', []) # [{'title': str, 'body': str, 'sources': [str]}]
state.setdefault('pending', None) # Future for the question currently waiting on the backend
state.setdefault('index_stale', False) # True after uploading files that aren't ingested yet
state.setdefault('uploader_key', 0) # Bumped to reset the file uploader after saving


# ---- Helpers ----

FRONT_MATTER = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def read_source(path):
    '''
        Returns (title, description, body) for a markdown source, using its front matter when present.
    '''
    text = path.read_text(encoding='utf-8', errors='replace')
    title, description, body = path.stem.replace('-', ' ').title(), '', text
    match = FRONT_MATTER.match(text)
    if match:
        body = text[match.end():]
        for line in match.group(1).splitlines():
            key, _, value = line.partition(':')
            value = value.strip().strip('"\'')
            if key.strip() == 'title' and value:
                title = value
            elif key.strip() == 'description':
                description = value
    return title, description, body

def list_sources():
    if not DATA_DIRECTORY.is_dir():
        return []
    return sorted(p for p in DATA_DIRECTORY.iterdir() if p.is_file() and p.suffix == '.md')

@st.cache_data(ttl=10, show_spinner=False)
def api_online():
    try:
        return requests.get(f'{API_URL}/openapi.json', timeout=1.5).ok
    except requests.RequestException:
        return False

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

@st.cache_resource
def executor():
    # Shared worker threads, so a slow answer doesn't block the page from rendering
    return ThreadPoolExecutor(max_workers=4)

def ask(question):
    '''
        Adds the question to the chat and sends it to the backend in the background.
    '''
    question = question.strip() if question else ''
    if not question or state.pending:
        return
    state.messages.append({'role': 'user', 'content': question})
    state.pending = executor().submit(api_call, 'POST', '/query', json={'user_query': question},
                                      timeout=QUERY_TIMEOUT)

@st.fragment(run_every=1.0)
def wait_for_answer():
    '''
        Shows the thinking indicator and checks once a second whether the answer has arrived.
        Only this fragment reruns while waiting, so the rest of the page stays usable.
    '''
    future = state.pending
    if future is None:
        return
    if not future.done():
        with st.container(key=f'msg_bot_{len(state.messages)}'):
            st.html('<span class="thinking">Thinking</span>')
        return
    result, error = future.result()
    if error:
        state.messages.append({'role': 'assistant', 'content': error, 'sources': [], 'error': True})
    else:
        state.messages.append({'role': 'assistant', 'content': result['response'], 'sources': result['sources']})
    state.pending = None
    st.rerun() # Full rerun to draw the answer with its citation chips

def notes_markdown():
    parts = []
    for note in state.notes:
        part = f"## {note['title']}\n\n{note['body']}"
        if note['sources']:
            part += f"\n\nSources: {', '.join(note['sources'])}"
        parts.append(part)
    return '\n\n---\n\n'.join(parts)


# ---- Dialogs ----

@st.dialog('Source', width='large')
def preview_source(name):
    path = DATA_DIRECTORY / name
    if not path.is_file():
        st.warning(f'{name} no longer exists in {DATA_DIRECTORY.name}/.')
        return
    title, description, body = read_source(path)
    st.markdown(f'### {title}')
    if description:
        st.caption(description)
    st.caption(f':material/description: {name}')
    st.divider()
    st.markdown(body)

@st.dialog('Clear the index?')
def confirm_clear():
    st.write('This removes every stored chunk from the vector database. '
             'Your files stay in the data folder, and **Rebuild index** brings them back.')
    left, right = st.columns(2)
    if left.button('Cancel', width='stretch'):
        st.rerun()
    if right.button('Clear index', type='primary', width='stretch'):
        _, error = api_call('DELETE', '/db', timeout=60)
        if error:
            st.error(error)
        else:
            state.index_stale = True
            st.toast('Index cleared', icon=':material/delete_sweep:')
            st.rerun()


# ---- Sidebar: sources and notes ----

online = api_online()
sources = list_sources()

with st.sidebar:
    st.html(f'''
    <div class="side-brand">
        <b>Notebook</b>
        <span class="status {"on" if online else "off"}"><span class="dot"></span>{"Online" if online else "Offline"}</span>
    </div>''')

    if st.button('New chat', icon=':material/edit_square:', width='stretch', disabled=not state.messages):
        state.messages, state.pending = [], None
        st.rerun()

    tab_sources, tab_notes = st.tabs([f'Sources ({len(sources)})', f'Notes ({len(state.notes)})'])

    with tab_sources:
        if not sources:
            st.html(f'<p class="muted">No .md files in <code>{DATA_DIRECTORY.name}/</code> yet. Upload some below.</p>')
        for i, path in enumerate(sources):
            title, description, _ = read_source(path)
            if st.button(title, key=f'src_{i}', icon=':material/article:', help=description or path.name,
                         width='stretch'):
                preview_source(path.name)

        st.divider()
        uploads = st.file_uploader('Add sources', type=['md'], accept_multiple_files=True,
                                   key=f'uploader_{state.uploader_key}', label_visibility='collapsed')
        if uploads:
            DATA_DIRECTORY.mkdir(exist_ok=True)
            for upload in uploads:
                (DATA_DIRECTORY / Path(upload.name).name).write_bytes(upload.getvalue())
            state.index_stale = True
            state.uploader_key += 1
            st.toast(f'Added {len(uploads)} source(s). Rebuild the index to include them.', icon=':material/upload_file:')
            st.rerun()

        if state.index_stale:
            st.html('<p class="muted">The index is out of date with your sources.</p>')
        if st.button('Rebuild index', icon=':material/sync:', width='stretch',
                     type='primary' if state.index_stale else 'secondary', disabled=not online):
            with st.spinner('Chunking and embedding sources...'):
                result, error = api_call('POST', '/ingest', timeout=INGEST_TIMEOUT)
            if error:
                st.error(error)
            else:
                state.index_stale = False
                st.toast(f"Index rebuilt with {result.get('chunks', '?')} chunks", icon=':material/check_circle:')
        if st.button('Clear index', icon=':material/delete_sweep:', width='stretch', disabled=not online):
            confirm_clear()

    with tab_notes:
        with st.container(horizontal=True, gap='small'):
            with st.popover('New note', icon=':material/edit_note:'):
                with st.form('new_note', clear_on_submit=True, border=False):
                    note_title = st.text_input('Title', placeholder='Untitled note')
                    note_body = st.text_area('Note', height=160)
                    if st.form_submit_button('Save', type='primary', width='stretch') and note_body.strip():
                        state.notes.insert(0, {'title': note_title.strip() or 'Untitled note',
                                               'body': note_body.strip(), 'sources': []})
                        st.rerun()
            if state.notes:
                st.download_button('Export', notes_markdown(), file_name='notes.md', mime='text/markdown',
                                   icon=':material/download:')

        if not state.notes:
            st.html('<p class="muted">Save answers from the chat or write your own notes. '
                    'Notes last for this browser session; export them to keep a copy.</p>')
        for i, note in enumerate(state.notes):
            with st.container(key=f'note_{i}'):
                top_left, top_right = st.columns([5, 1], vertical_alignment='top')
                top_left.html(f'<div class="note-title">{html.escape(note["title"])}</div>')
                if top_right.button('', key=f'del_note_{i}', icon=':material/delete:', type='tertiary',
                                    help='Delete note'):
                    state.notes.pop(i)
                    st.rerun()
                st.markdown(note['body'])
                if note['sources']:
                    st.html(f'<div class="note-sources">Sources: {html.escape(", ".join(note["sources"]))}</div>')


# ---- Chat ----

composer = dict(key='chat_box', on_submit=lambda: ask(state.chat_box),
                disabled=bool(state.pending) or not online,
                placeholder='Ask about your sources...' if online else 'Backend offline. Start uvicorn main:app')

if not state.messages:
    # Empty state: greeting with the composer centred underneath
    st.html(f'''
    <div class="greeting">
        <h1>What would you like to know?</h1>
        <p>Answers come only from your {len(sources)} source{"s" if len(sources) != 1 else ""}, with citations.</p>
    </div>''')
    with st.container(key='composer_hero'):
        st.chat_input(**composer)
    suggestions = [f'Summarize "{read_source(p)[0]}"' for p in sources[:3]]
    if suggestions:
        with st.container(key='suggestions', horizontal=True, horizontal_alignment='center', gap='small'):
            for i, suggestion in enumerate(suggestions):
                st.button(suggestion, key=f'suggest_{i}', on_click=ask, args=(suggestion,))
else:
    for i, message in enumerate(state.messages):
        if message['role'] == 'user':
            with st.container(key=f'msg_user_{i}'):
                st.markdown(message['content'])
            continue

        with st.container(key=f'msg_bot_{i}'):
            if message.get('error'):
                st.html(f'<span class="msg-error">{html.escape(message["content"])}</span>')
                continue
            st.markdown(message['content'])
            with st.container(horizontal=True, gap='small'):
                for j, name in enumerate(message['sources']):
                    if st.button(Path(name).stem, key=f'chip_{i}_{j}', icon=':material/description:'):
                        preview_source(name)
                if st.button('Save to notes', key=f'save_{i}', icon=':material/bookmark_add:'):
                    question = state.messages[i - 1]['content'] if i > 0 else 'Saved answer'
                    state.notes.insert(0, {'title': question, 'body': message['content'],
                                           'sources': message['sources']})
                    st.toast('Saved to notes', icon=':material/bookmark:')
                    st.rerun()

    if state.pending:
        wait_for_answer()

    # Called at the top level, so Streamlit pins it to the bottom of the page
    st.chat_input(**composer)
