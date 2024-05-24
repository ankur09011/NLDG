PAGE_1 = """
# Natural Language Dashboard Generation using **Taipy**{: .color-primary} **GUI**{: .color-primary}

<|layout|columns=1|
<|
**Your Query:** <|{text}|>

**Enter query for smart generation:**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>
|>


<|layout|columns=1||>
<|Query Table with matches |expandable|
<|{dataframe}|table|width=100%|number_format=%.2f|rebuild||>
|>
|>

"""


extra = """
<|layout|columns=1 1 1|
## Positive <|{np.mean(dataframe['Score Pos'])}|text|format=%.2f|raw|>

## Neutral <|{np.mean(dataframe['Score Neu'])}|text|format=%.2f|raw|>

## Negative <|{np.mean(dataframe['Score Neg'])}|text|format=%.2f|raw|>
|>


"""