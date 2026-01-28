import os

if os.environ.get("VERCEL") != "1":
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
