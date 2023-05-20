import streamlit as st
import pandas as pd
from predict import get_review_prediction
PAGE_CONFIG = {"page_title":"Fallout 76 Review Aspect Sentiment Analysis","page_icon":":video_game:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

def process_result(pred_res):
	res = pred_res
	for model in res.keys():
		temp_dict = {}
		for aspect in res[model].keys():
			for label in res[model][aspect][0].keys():
				temp_dict[(aspect,label)] = res[model][aspect][0][label] 
		res[model] = temp_dict
	return pd.DataFrame.from_dict(res)

def main():
	st.title("Fallout 76 Review Aspect Sentiment Analysis")
	review_text = st.text_area("Review Text",height=250)
	if st.button("Predict"):
		pred_res = get_review_prediction([review_text])
		pred_df = process_result(pred_res)
		pred_label = {}
		for aspect in pred_df.index.get_level_values(0).unique():
			temp_df = pred_df.loc[aspect].idxmax()
			pred_label[aspect] = temp_df['heuristic']
		st.dataframe(pd.DataFrame(pred_label.values(), columns=['Predicted Label'], index=pred_label.keys()))
		with st.expander("Model Scores"):
			st.dataframe(pred_df)	
if __name__ == '__main__':
	main()