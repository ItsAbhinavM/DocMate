import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

def count_documents_by_type() -> Dict[str, int]:
    """Count documents in the uploads directory by file extension."""
    if not os.path.exists("uploads"):
        return {}
    
    files = os.listdir("uploads")
    extensions = [os.path.splitext(f)[1].lower() for f in files if os.path.isfile(os.path.join("uploads", f))]
    return dict(Counter(extensions))

def estimate_token_counts(docs) -> Dict[str, Any]:
    """Estimate token counts for each document."""
    if not docs:
        return {"total_tokens": 0, "avg_tokens_per_doc": 0, "doc_token_counts": []}
    
    # Rough estimation: ~1.3 tokens per word
    def estimate_tokens(text):
        words = len(re.findall(r'\b\w+\b', text))
        return int(words * 1.3)
    
    doc_token_counts = [(doc.metadata.get('source', 'Unknown'), estimate_tokens(doc.page_content)) for doc in docs]
    total_tokens = sum(count for _, count in doc_token_counts)
    
    return {
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": total_tokens / len(docs) if docs else 0,
        "doc_token_counts": doc_token_counts
    }

def document_length_stats(docs) -> Dict[str, Any]:
    """Calculate statistics about document lengths."""
    if not docs:
        return {"avg_length": 0, "max_length": 0, "min_length": 0}
    
    lengths = [len(doc.page_content) for doc in docs]
    
    return {
        "avg_length": np.mean(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "median_length": np.median(lengths),
        "std_dev_length": np.std(lengths)
    }

def get_content_word_frequency(docs, top_n=20) -> List[tuple]:
    """Get the most common words across all documents."""
    if not docs:
        return []
    
    # Combine all text
    all_text = " ".join([doc.page_content for doc in docs])
    
    # Remove basic stopwords
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    # Extract words and count them
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    words = [word for word in words if word not in stopwords]
    
    return Counter(words).most_common(top_n)

def generate_file_type_chart(stats, output_path="static/filetype_stats.png"):
    """Generate a pie chart of document types."""
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not stats:
        stats = {'.none': 1}  # Default if no stats
    
    # Clean extension names for display
    labels = [ext[1:] if ext.startswith('.') else ext for ext in stats.keys()]
    
    plt.figure(figsize=(8, 6))
    plt.pie(stats.values(), labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Document Types Distribution')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_token_count_chart(token_stats, output_path="static/token_stats.png"):
    """Generate a bar chart of token counts per document."""
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not token_stats.get("doc_token_counts"):
        return None
    
    # Extract data for chart
    sources = [src.split('/')[-1] if isinstance(src, str) else "Document" for src, _ in token_stats["doc_token_counts"]]
    counts = [count for _, count in token_stats["doc_token_counts"]]
    
    # Limit to top 15 documents if we have too many
    if len(sources) > 15:
        indices = np.argsort(counts)[-15:]
        sources = [sources[i] for i in indices]
        counts = [counts[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(sources, counts)
    plt.xlabel('Estimated Token Count')
    plt.title('Estimated Tokens per Document')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_word_frequency_chart(word_freq, output_path="static/word_freq_stats.png"):
    """Generate a bar chart of most common words."""
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not word_freq:
        return None
    
    words = [word for word, _ in word_freq]
    counts = [count for _, count in word_freq]
    
    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1])  # Reverse to put highest at top
    plt.xlabel('Frequency')
    plt.title('Most Common Words in Documents')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_all_statistics(docs) -> Dict[str, Any]:
    """Generate comprehensive statistics for documents."""
    file_type_stats = count_documents_by_type()
    token_stats = estimate_token_counts(docs)
    length_stats = document_length_stats(docs)
    word_freq = get_content_word_frequency(docs)
    
    # Generate charts
    file_type_chart = generate_file_type_chart(file_type_stats)
    token_count_chart = generate_token_count_chart(token_stats)
    word_freq_chart = generate_word_frequency_chart(word_freq)
    
    return {
        "file_types": file_type_stats,
        "token_stats": token_stats,
        "length_stats": length_stats,
        "top_words": word_freq,
        "charts": {
            "file_type_chart": file_type_chart,
            "token_count_chart": token_count_chart,
            "word_freq_chart": word_freq_chart
        }
    }
