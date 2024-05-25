PAGE_1 = """
# Natural Language Dashboard Generation 

<|layout|columns=1||>
**Your Query:** <|{text}|>

**Enter query for smart generation:**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>


<|layout|columns=1||>
<|Query Table with matches |expandable||>
<|{dataframe}|table|width=100%|number_format=%.2f|rebuild||>


**last query softmax score**
<|layout|columns=1||>
<|{dataframe1}|table|width=100%|number_format=%.2f|rebuild||>

<|layout|columns=1 1 1||>
## Positive <|{np.mean(dataframe1['Score Pos'])}|text|format=%.2f|raw|>

## Neutral <|{np.mean(dataframe1['Score Neu'])}|text|format=%.2f|raw|>

## Negative <|{np.mean(dataframe1['Score Neg'])}|text|format=%.2f|raw|>

"""


extra = """
<|layout|columns=1 1 1||>
## Positive <|{np.mean(dataframe['Score Pos'])}|text|format=%.2f|raw|>

## Neutral <|{np.mean(dataframe['Score Neu'])}|text|format=%.2f|raw|>

## Negative <|{np.mean(dataframe['Score Neg'])}|text|format=%.2f|raw|>
|>


"""

PAGE_RAW_DB = """
# Raw Data used

<|layout|columns=1||>
<|Raw queries mapping for Semantic Search |expandable||>
<|{raw_text}|table|width=100%|number_format=%.2f|rebuild|page_size=50||>

<|layout|columns=1||>
<|Raw sales transaction data |expandable||>
<|{raw_query_df}|table|width=100%|number_format=%.2f|rebuild|page_size=50||>


<|layout|columns=1||>
<|Query Results |expandable||>
<|{dataframe}|table|width=100%|number_format=%.2f|rebuild|page_size=50||>


"""
