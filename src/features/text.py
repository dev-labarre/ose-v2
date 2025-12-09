"""
Text processing: FastText → PCA(10).
Builds combined_titles from pre-t0 articles, trains FastText, extracts embeddings,
and applies PCA(10) to reduce to 10 components.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import os

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("⚠️  fastText not available, will use TF-IDF fallback")

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

T0 = pd.Timestamp('2023-01-01', tz='UTC')


def build_combined_titles(df_articles: pd.DataFrame, siren: str) -> str:
    """
    Build combined_titles from articles for a company (pre-t0 only).
    
    Args:
        df_articles: DataFrame with articles
        siren: SIREN identifier
        
    Returns:
        Combined titles string
    """
    company_articles = df_articles[
        (df_articles['siren'] == siren) & 
        (df_articles.get('publishedAt_parsed', pd.Series([pd.NaT] * len(df_articles))) < T0)
    ]
    
    if len(company_articles) == 0:
        return ''
    
    titles = company_articles['title'].fillna('').astype(str)
    combined = ' '.join(titles.tolist())
    return combined


def train_fasttext_embeddings(texts: list, embedding_dim: int = 100) -> object:
    """
    Train FastText model on texts.
    
    Args:
        texts: List of text strings
        embedding_dim: Embedding dimension
        
    Returns:
        Trained FastText model
    """
    if not FASTTEXT_AVAILABLE:
        raise ImportError("fastText not available")
    
    # Create temporary file for FastText training
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for text in texts:
            if text and text.strip():
                f.write(text.strip() + '\n')
        temp_file = f.name
    
    try:
        # Train FastText model
        model = fasttext.train_unsupervised(
            temp_file,
            model='skipgram',
            dim=embedding_dim,
            epoch=5,
            minCount=2,
            verbose=1
        )
        return model
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def extract_sentence_embedding(model: object, text: str, embedding_dim: int = 100) -> np.ndarray:
    """
    Extract sentence embedding from FastText model.
    
    Args:
        model: Trained FastText model
        text: Text string
        embedding_dim: Embedding dimension
        
    Returns:
        Sentence embedding vector
    """
    if not text or not text.strip():
        return np.zeros(embedding_dim)
    
    words = text.strip().split()
    word_embeddings = []
    
    for word in words:
        try:
            vec = model.get_word_vector(word)
            word_embeddings.append(vec)
        except:
            continue
    
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)


def process_text_features(df_features: pd.DataFrame, df_articles: pd.DataFrame,
                          output_path: Path) -> pd.DataFrame:
    """
    Process text features: FastText → PCA(10).
    
    Args:
        df_features: Features DataFrame
        df_articles: Articles DataFrame (pre-t0 filtered)
        output_path: Path to save text ablation report
        
    Returns:
        Features DataFrame with PCA(10) text features added
    """
    print("="*80)
    print("TEXT PROCESSING: FastText → PCA(10)")
    print("="*80)
    
    # Build combined_titles for each company (pre-t0 only)
    print("\n📝 Building combined_titles from pre-t0 articles...")
    
    if 'publishedAt' in df_articles.columns:
        df_articles['publishedAt_parsed'] = pd.to_datetime(
            df_articles['publishedAt'], errors='coerce'
        )
        df_articles_pre_t0 = df_articles[df_articles['publishedAt_parsed'] < T0].copy()
    else:
        df_articles_pre_t0 = df_articles.copy()
    
    print(f"  ✓ Found {len(df_articles_pre_t0)} pre-t0 articles")
    
    # Build combined titles
    combined_titles = []
    for siren in df_features['siren']:
        titles = df_articles_pre_t0[df_articles_pre_t0['siren'] == siren]['title']
        combined = ' '.join(titles.fillna('').astype(str).tolist())
        combined_titles.append(combined)
    
    df_features['combined_titles'] = combined_titles
    print(f"  ✓ Built combined_titles for {len(df_features)} companies")
    
    # Extract FastText embeddings
    print("\n🔤 Extracting FastText embeddings...")
    
    texts = df_features['combined_titles'].fillna('').tolist()
    
    use_fasttext = FASTTEXT_AVAILABLE
    if use_fasttext:
        try:
            model = train_fasttext_embeddings(texts, embedding_dim=100)
            print("  ✓ FastText model trained")
            
            embeddings = []
            for text in texts:
                emb = extract_sentence_embedding(model, text, embedding_dim=100)
                embeddings.append(emb)
            
            embeddings_array = np.array(embeddings)
            print(f"  ✓ Extracted embeddings: shape {embeddings_array.shape}")
        except Exception as e:
            print(f"  ⚠️  FastText failed: {e}, using TF-IDF fallback")
            use_fasttext = False
    
    if not use_fasttext:
        # Fallback to TF-IDF
        print("  Using TF-IDF fallback...")
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        embeddings_array = vectorizer.fit_transform(texts).toarray()
        print(f"  ✓ TF-IDF embeddings: shape {embeddings_array.shape}")
    
    # Apply PCA(10)
    print("\n📊 Applying PCA(10)...")
    pca = PCA(n_components=10, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings_array)
    
    print(f"  ✓ PCA applied: {embeddings_array.shape[1]} → 10 components")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Add PCA components as features
    for i in range(10):
        df_features[f'text_pca_{i}'] = pca_embeddings[:, i]
    
    print(f"  ✓ Added 10 PCA text features")
    
    # Generate ablation report
    ablation_report = {
        'text_processing': {
            'method': 'FastText' if use_fasttext else 'TF-IDF',
            'embedding_dim': embeddings_array.shape[1],
            'pca_components': 10,
            'explained_variance': float(pca.explained_variance_ratio_.sum()),
            'n_companies_with_text': int((df_features['combined_titles'].str.len() > 0).sum())
        },
        'text_features': [f'text_pca_{i}' for i in range(10)]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_report, f, indent=2, default=str)
    
    print(f"  ✓ Text ablation report saved to {output_path}")
    
    return df_features


def main():
    """Main function."""
    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / 'data' / 'features' / 'features.parquet'
    articles_path = project_root / 'data' / 'raw_json' / '09_articles.json'
    output_path = project_root / 'reports' / 'text_ablation.json'
    
    if not features_path.exists():
        print(f"⚠️  Features file not found: {features_path}")
        return
    
    df_features = pd.read_parquet(features_path)
    
    # Load articles
    import json
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    df_articles = pd.DataFrame(articles_data)
    
    df_features = process_text_features(df_features, df_articles, output_path)
    
    # Save updated features
    df_features.to_parquet(features_path, index=False)
    print(f"\n✓ Updated features saved to {features_path}")


if __name__ == "__main__":
    main()
