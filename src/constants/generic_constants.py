PAGE_1 = """
# Natural Language Dashboard Generation using **Taipy**{: .color-primary} **GUI**{: .color-primary}

<|layout|columns=1|
<|
**Your Query:** <|{text}|>

**Enter query for smart generation:**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>
|>


<|layout|columns=1|
<|Query Table with matches |expandable|
<|{dataframe}|table|width=100%|number_format=%.2f|>
|>
|>

<|{dataframe}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|>
"""


extra = """
<|layout|columns=1 1 1|
## Positive <|{np.mean(dataframe['Score Pos'])}|text|format=%.2f|raw|>

## Neutral <|{np.mean(dataframe['Score Neu'])}|text|format=%.2f|raw|>

## Negative <|{np.mean(dataframe['Score Neg'])}|text|format=%.2f|raw|>
|>


"""