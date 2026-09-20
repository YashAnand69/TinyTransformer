"""Compatibility helpers for the versioned educational curriculum."""
from pathlib import Path
from curriculum import build_splits

def get_full_corpus() -> str:
    return '\n\n'.join(doc['text'] for doc in build_splits()['train'])

def save_corpus_to_file(path: str) -> str:
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    text=get_full_corpus()
    target.write_text(text,encoding='utf-8')
    return text

if __name__=='__main__':
    text=get_full_corpus()
    print(f'Training corpus: {len(text):,} characters, {len(text.split()):,} words.')
