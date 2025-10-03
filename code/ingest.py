import io
import re
import zipfile
import requests
import frontmatter

from minsearch import Index

from logger import logger

def read_repo_data(repo_owner, repo_name):
    url = f'https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not (filename.endswith('.md') or filename.endswith('.mdx')):
            continue

        with zf.open(file_info) as f_in:
            content = f_in.read()
            post = frontmatter.loads(content)
            data = post.to_dict()

            _, filename_repo = file_info.filename.split('/', maxsplit=1)
            data['filename'] = filename_repo
            repository_data.append(data)

    zf.close()

    return repository_data


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i:i+size]
        result.append({'start': i, 'content': batch})
        if i + size > n:
            break

    return result


def chunk_documents_with_sliding_window(docs, size=2000, step=1000):
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        doc_chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in doc_chunks:
            chunk.update(doc_copy)
        chunks.extend(doc_chunks)

    return chunks


def split_markdown_by_level(text, level=2):
    """
    Split markdown text by a specific header level.
    
    :param text: Markdown text as a string
    :param level: Header level to split on
    :return: List of sections as strings
    """
    # This regex matches markdown headers
    # For level 2, it matches lines starting with "## "
    header_pattern = r'^(#{' + str(level) + r'} )(.+)$'
    pattern = re.compile(header_pattern, re.MULTILINE)

    # Split and keep the headers
    parts = pattern.split(text)
    
    sections = []
    for i in range(1, len(parts), 3):
        # We step by 3 because regex.split() with
        # capturing groups returns:
        # [before_match, group1, group2, after_match, ...]
        # here group1 is "## ", group2 is the header text
        header = parts[i] + parts[i+1]  # "## " + "Title"
        header = header.strip()

        # Get the content after this header
        content = ""
        if i+2 < len(parts):
            content = parts[i+2].strip()

        if content:
            section = f'{header}\n\n{content}'
        else:
            section = header
        sections.append(section)
    
    return sections


def chunk_documents_with_markdown_sections(docs, level=2):
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        doc_chunks = split_markdown_by_level(doc_content, level=level)
        for chunk in doc_chunks:
            chunk_data = {'content': chunk}
            chunk_data.update(doc_copy)
            chunks.append(chunk_data)

    return chunks


def chunk_documents(docs, method='sliding_window', **kwargs):
    if method == 'sliding_window':
        size = kwargs.get('size', 2000)
        step = kwargs.get('step', 1000)
        return chunk_documents_with_sliding_window(docs, size=size, step=step)
    elif method == 'markdown_sections':
        level = kwargs.get('level', 2)
        return chunk_documents_with_markdown_sections(docs, level=level)
    else:
        raise ValueError(f"Unknown chunking method: {method}")


def index_data(
        repo_owner,
        repo_name,
        filter=None,
        chunk=True,
        chunk_method='sliding_window',
        chunking_params=None,
    ):
    docs = read_repo_data(repo_owner, repo_name)
    logger.info(f"Fetched {len(docs)} documents from the repository.")

    if filter is not None:
        docs = [doc for doc in docs if filter(doc)]

    if chunk:
        logger.info(f"Chunking documents using {chunk_method}...")
        if chunk_method == 'sliding_window' and chunking_params is None:
            chunking_params = {'size': 2000, 'step': 1000}
        elif chunk_method == 'markdown_sections' and chunking_params is None: 
            chunking_params = {'level': 2}
            
        docs = chunk_documents(docs, **chunking_params)
        logger.info(f"Total chunks created: {len(docs)}")

    logger.info("Indexing documents...")
    index = Index(
        text_fields=["content", "filename"],
    )

    index.fit(docs)
    return index