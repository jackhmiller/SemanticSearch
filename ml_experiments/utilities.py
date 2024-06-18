import itertools
import pandas as pd

def df_col_to_sentences(features: list[str], data: pd.DataFrame) -> list[str]:
	if len(features) > 1:
		sentence_pairs = data[features].values.tolist()
		sentences = [' '.join(i) for i in sentence_pairs]
	else:
		sentences = data[features].values.tolist()
		sentences = list(itertools.chain.from_iterable(sentences))

	return sentences