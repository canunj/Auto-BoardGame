import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#Gender marker plot
def marker_count(pat,df=None):
  if df is not None:
    cts = df['description'].str.count(re.compile(pat))
    lens = [len(x.split()) for x in df['description']]
    return [a/b for a,b in zip(list(cts),lens)]
  else:
    cts = pd.concat([val_comp['.4 Temp'], val_comp['.5 Temp'], val_comp['.6 Temp'], val_comp['.7 Temp'], val_comp['.8 Temp']]).str.count(re.compile(pat))
    lens = [len(x.split()) for x in pd.concat([val_comp['.4 Temp'], val_comp['.5 Temp'], val_comp['.6 Temp'], val_comp['.7 Temp'], val_comp['.8 Temp']])]
    return [a/b for a,b in zip(list(cts),lens)]

hold = pd.DataFrame({"Ind":np.arange(100)})

hold["his"] = marker_count("\\b[Hh][Ii][Ss]\\b")
hold["he"] = marker_count("\\b[Hh][Ee]\\b")
hold["him"] = marker_count("\\b[Hh][Ii][Mm]\\b")
hold["hers"] = marker_count("\\b[Hh][Ee][Rr][Ss]\\b")
hold["her"] = marker_count("\\b[Hh][Ee][Rr]\\b")
hold["she"] = marker_count("\\b[Ss][Hh][Ee]\\b")
slim_df["his"] = marker_count("\\b[Hh][Ii][Ss]\\b",slim_df)
slim_df["he"] = marker_count("\\b[Hh][Ee]\\b",slim_df)
slim_df["him"] = marker_count("\\b[Hh][Ii][Mm]\\b",slim_df)
slim_df["hers"] = marker_count("\\b[Hh][Ee][Rr][Ss]\\b",slim_df)
slim_df["her"] = marker_count("\\b[Hh][Ee][Rr]\\b",slim_df)
slim_df["she"] = marker_count("\\b[Ss][Hh][Ee]\\b",slim_df)
df1 = hold[["he","she"]]
df2 = slim_df[["he","she"]]
df = df1.join(df2, lsuffix="_first", rsuffix=("_second"))

def show_dist(cols):
  
  subs = make_subplots(rows=2, cols=2, vertical_spacing=0.2,subplot_titles=("Source M", "Source F", "Gen M", "Gen F"))
  col = 1
  row = 1

  for x in cols:
    new_df = x.value_counts().rename_axis(x.name).reset_index(name='counts')
    fig = px.bar(new_df, x=x.name, y="counts", color="counts", title="----------") 
    subs = subs.add_trace(fig.data[0], row=row, col=col)
    if col > row:
      row += 1
      col = 1
    elif col == row:
      col += 1

  #subs['layout']['xaxis4'].update(range=[0,0.5])
  subs.show()

fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                    vertical_spacing=0.05,subplot_titles=("BGG He", "BGG She", "Generated He", "Generated She"),
                    x_title="Proportional Word Appearance",y_title="Description Count")

# set the x-axis range
fig.update_xaxes(range=[0, 0.025])
fig.update_yaxes(range=[0,100])


    # create a histogram with 20 bins and normalized to show density
fig.add_trace(go.Histogram(x=df2['he'],name="BGG He", marker=dict(color="MediumSlateBlue")), row=1, col=1)
fig.add_trace(go.Histogram(x=df1['he'],name="Generated He", marker=dict(color="DarkSeaGreen")), row=2, col=1)
fig.add_trace(go.Histogram(x=df2['she'],name="BGG She",marker=dict(color="MediumSlateBlue")), row=1, col=2)
fig.add_trace(go.Histogram(x=df1['she'],name="Generated She",marker=dict(color="DarkSeaGreen")), row=2, col=2)

# update layout
fig.update_layout(height=800, width=800,
                  title_text="Comparison of Per Description Appearance Rate for He/She Markers",
                  title_x=0.5,
                  font_family="Inter",
                  showlegend=False)


# display plot
fig.show()


# tuning params
df = pd.read_csv('/content/final_tuning.csv')
fig = go.Figure()

fig.add_scatter(x=df['vals'], y=df['precision'], name = "Precision")
fig.add_scatter(x=df['vals'], y=df['recall'], name = "Recall")
fig.add_scatter(x=df['vals'], y=df['f1'], name = "F1")
fig.add_scatter(x=df['vals'], y=df['rouge1'], name = "Rouge - Unigram")
fig.add_scatter(x=df['vals'], y=df['rouge2'], name = "Rouge - Bigram")
fig.add_scatter(x=df['vals'], y=df['rougeL'], name = "Rouge - Longest")
fig.add_scatter(x=df['vals'], y=df['bleurt'], name = "Bleurt")

# update layout
fig.update_layout(height=800, width=800,
                  xaxis_title="Profile Parameters (Temp, Presence)",
                  yaxis_title="Score",
                  legend_title="Evaluation Metric",
                  title_text="Fine-Tuned Analysis",
                  title_x=0.5,
                  font_family="Inter",
                  font_size=14)


# display plot
fig.show()
fig.write_image("Gen_Profiles.jpeg")

#memorization evaluation
df = pd.read_parquet("/content/mem_test.parquet.gzip")
names = []
for name in df['name']:
  if len(name) > 20:
    names.append(name[0:17]+"...")
  else:
    names.append(name)
df['name'] = names

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True,
                    vertical_spacing=0.05,subplot_titles=(".5 Temp, .7 Presence", ".4 Temp, .6 Presence", ".5 Temp, .8 Presence"),y_title="Evaluation Score")


fig.add_trace(go.Scatter(x=df['name'], y=df['.5.7 precision'], name = "Precision", legendgroup="group",mode="markers",marker=dict(color="#1f77b4",size=5)),row=1,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.7 recall'], name = "Recall",legendgroup="group",mode="markers",marker=dict(color="#ff7f0e",size=5)),row=1,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.7 f1'], name = "F1",legendgroup="group",mode="markers",marker=dict(color="#2ca02c",size=5)),row=1,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.7 r1'], name = "Rouge - Unigram",legendgroup="group",mode="markers",marker=dict(color="#d62728",size=5)),row=1,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.7 r2'], name = "Rouge - Bigram",legendgroup="group",mode="markers",marker=dict(color="#9467bd",size=5)),row=1,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.7 rL'], name = "Rouge - Longest",legendgroup="group",mode="markers",marker=dict(color="#8c564b",size=5)),row=1,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.7 bleurt'], name = "Bleurt",legendgroup="group",mode="markers",marker=dict(color="#e377c2",size=5)),row=1,col=1)

fig.add_trace(go.Scatter(x=df['name'], y=df['.4.6 precision'], name = "Precision",showlegend=False,mode="markers",marker=dict(color="#1f77b4",size=5)),row=2,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.4.6 recall'], name = "Recall",showlegend=False,mode="markers",marker=dict(color="#ff7f0e",size=5)),row=2,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.4.6 f1'], name = "F1",showlegend=False,mode="markers",marker=dict(color="#2ca02c",size=5)),row=2,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.4.6 r1'], name = "Rouge - Unigram",showlegend=False,mode="markers",marker=dict(color="#d62728",size=5)),row=2,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.4.6 r2'], name = "Rouge - Bigram",showlegend=False,mode="markers",marker=dict(color="#9467bd",size=5)),row=2,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.4.6 rL'], name = "Rouge - Longest",showlegend=False,mode="markers",marker=dict(color="#8c564b",size=5)),row=2,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.4.6 bleurt'], name = "Bleurt",showlegend=False,mode="markers",marker=dict(color="#e377c2",size=5)),row=2,col=1)

fig.add_trace(go.Scatter(x=df['name'], y=df['.5.8 precision'], name = "Precision",showlegend=False,mode="markers",marker=dict(color="#1f77b4",size=5)),row=3,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.8 recall'], name = "Recall",showlegend=False,mode="markers",marker=dict(color="#ff7f0e",size=5)),row=3,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.8 f1'], name = "F1",showlegend=False,mode="markers",marker=dict(color="#2ca02c",size=5)),row=3,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.8 r1'], name = "Rouge - Unigram",showlegend=False,mode="markers",marker=dict(color="#d62728",size=5)),row=3,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.8 r2'], name = "Rouge - Bigram",showlegend=False,mode="markers",marker=dict(color="#9467bd",size=5)),row=3,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.8 rL'], name = "Rouge - Longest",showlegend=False,mode="markers",marker=dict(color="#8c564b",size=5)),row=3,col=1)
fig.add_trace(go.Scatter(x=df['name'], y=df['.5.8 bleurt'], name = "Bleurt",showlegend=False,mode="markers",marker=dict(color="#e377c2",size=5)),row=3,col=1)

# update layout
fig.update_layout(height=1200, width=1600,
                  legend_title="Evaluation Metric",
                  title_text="Training Data Memorization Metrics",
                  title_x=0.5,
                  font_family="Inter",
                  margin=dict(b=170))

fig.update_xaxes(tickmode='linear',tickfont = dict(size=12))
fig.add_annotation(xref='paper',yref='paper',text="Game",showarrow=False,font_size=16,y=-0.15)

# display plot
fig.show()
fig.write_image("Train_Analysis.jpeg")