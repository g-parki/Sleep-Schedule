from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import BoxAnnotation, Range1d, PanTool, WheelZoomTool, ResetTool, Label, DatetimeTickFormatter
from bokeh.transform import jitter
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def convert_timezone(timestamp):
    ts = pd.to_datetime(timestamp)
    datetime_obj = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    #return pytz.utc.localize(datetime_obj)
    return pytz.utc.localize(datetime_obj).astimezone(pytz.timezone('US/Pacific')).replace(tzinfo=None)

convert_timezone_np = np.vectorize(convert_timezone)

def live_prediction_graph(data_source):
    """Returns script, div of plot for a live readout of the neural network inference"""

    #Build plot for live prediction reading
    p = figure(
        plot_height=60,
        sizing_mode= 'scale_width',
        x_range= Range1d(start=-1.05, end=1.05, bounds=(-1.05,1.05)),
        y_range= Range1d(start=-0.1, end=0.1, bounds=(-0.25,0.25)),
        tools= '',
        toolbar_location= None,
        min_border= 0,
    )

    #Shaded backgrounds
    red_polygon = BoxAnnotation(
        right= 0,
        fill_color="crimson",
        fill_alpha=0.05,
    )
    blue_polygon = BoxAnnotation(
        left= 0,
        fill_color="dodgerblue",
        fill_alpha=0.05,
    )
    #Bold "end zones"
    red_polygon_end = BoxAnnotation(
        right= -1,
        fill_color="crimson",
        fill_alpha=.5,
    )
    blue_polygon_end = BoxAnnotation(
        left= 1,
        fill_color="dodgerblue",
        fill_alpha=.5,
    )

    #Labels
    left_annotation = Label(x=-.93, y=-.065, text="Empty", text_align='left', text_font_size = '12px')
    center_annotation = Label(x=0, y=-.065, text= "Live Prediction", text_align='center', text_font_size = '14px', text_font_style= 'bold')
    right_annotation = Label(x=.93, y=-.065, text= "Baby", text_align='right', text_font_size = '12px')
    
    #Add all of the above
    p.add_layout(red_polygon_end)
    p.add_layout(blue_polygon_end)
    p.add_layout(red_polygon)
    p.add_layout(blue_polygon)
    p.add_layout(left_annotation)
    p.add_layout(center_annotation)
    p.add_layout(right_annotation)
    
    p.scatter('x', 'y',
        source=data_source,
        color= 'black',
        size= 10,
        fill_alpha= .3
        )
    p.yaxis.visible = False
    p.xaxis.visible = False
    p.ygrid.visible = False

    return components(p)

def model_performance_graph(baby_df, nobaby_df):
    """Returns script, div of a plot of a model's predictions for the entire dataset"""

    #HTML tooltip which references the PhotoURL in the dataframe
    tools = [PanTool(), WheelZoomTool(maintain_focus= False), ResetTool()]
    TOOLTIPS = '<div><img src= "@PhotoURL"><p>@FileName</p></div>'

    baby_src = ColumnDataSource(data= baby_df)
    nobaby_src = ColumnDataSource(data= nobaby_df)
    JITTER_RADIUS_X = .11
    JITTER_RADIUS_Y = 1
    DOT_SIZE = 10
    DOT_ALPHA = .1
    BACK_ALPHA = .05

    p = figure(
        tooltips= TOOLTIPS,
        tools=tools,
        toolbar_location="below",
        toolbar_sticky=False,
        active_scroll= tools[1],
        x_axis_location="above",
        y_range=Range1d(start=-.02, end=1.08, bounds=(-.25,1.25)),
        x_range=Range1d(start=-(1+1.5*(JITTER_RADIUS_X)), end=1+1.5*JITTER_RADIUS_X, bounds=(-1.25,1.25))
    )

    #Shaded background
    red_polygon = BoxAnnotation(
        right= 0,
        fill_color= "crimson",
        fill_alpha= BACK_ALPHA,
    )
    blue_polygon = BoxAnnotation(
        left= 0,
        fill_color="dodgerblue",
        fill_alpha= BACK_ALPHA,
    )
    p.add_layout(red_polygon)
    p.add_layout(blue_polygon)

    #Data points, jittered to separate the dots
    p.circle(
        jitter('NoBabyLikeliness', JITTER_RADIUS_X),
        jitter('y', JITTER_RADIUS_Y),
        source= nobaby_src,
        size=DOT_SIZE,
        color='red', alpha=DOT_ALPHA,
        legend_label="Photos without baby  "
    )
    p.circle(
        jitter('BabyLikeliness', JITTER_RADIUS_X),
        jitter('y', JITTER_RADIUS_Y),
        source= baby_src,
        size=DOT_SIZE,
        color='blue', alpha=DOT_ALPHA,
        legend_label="Photos with baby   "
    )

    p.sizing_mode = 'scale_both'

    #Axis config
    p.yaxis.visible = False
    p.ygrid.visible = False
    p.xaxis.axis_label = '<- Predicted to Not Have Baby    Predicted to Have Baby ->      '
    p.xaxis.axis_label_text_font_size = '8pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.major_label_text_font_size = '0pt'  #turn off x-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt' #turn off y-axid tick labels
    
    #Legend config
    p.legend.location = "top_center"
    p.legend.click_policy="hide"
    p.legend.label_text_font_size = '8pt'
    p.legend.label_text_font_style = 'normal'
    p.legend.orientation='vertical'
    p.legend.glyph_height= 20
    p.legend.background_fill_alpha = 0.7
    p.legend.border_line_width = 1

    return components(p)

def bedtime_graph(sourceDF, fillsourceDF):
    """Returns script, div of a plot of automatic readings over last 24 hours"""
    
    tools = [PanTool(), WheelZoomTool(maintain_focus= False), ResetTool()]
    TOOLTIPS = '<div"><img src= "@PhotoURL"><p>@FileName</p></div>'
    DOT_SIZE = 10
    DOT_ALPHA = .20

    source = ColumnDataSource(
        data= dict(
            date= convert_timezone_np(sourceDF.index),
            value= sourceDF['nap_time'],
            PhotoURL= sourceDF['PhotoURL'],
            FileName= sourceDF['FileName']
        )
    )

    fillsource = ColumnDataSource(
        data= dict(date= convert_timezone_np(fillsourceDF.index),
        value=fillsourceDF['nap_time'])
    )

    p = figure(
        x_axis_type='datetime',
        tools= tools,
        tooltips= TOOLTIPS,
        toolbar_location= None,
        toolbar_sticky= False,
        active_scroll= tools[1],
        x_range=Range1d(start=datetime.now() - timedelta(hours=24),
            end= datetime.now(),
            bounds= (
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
        ),
        y_range=Range1d(start=-.07, end=1.07, bounds=(-.07,1.07)),
        plot_height=200,
        outline_line_color= None
    )

    p.circle('date', jitter('value', .04), source= source, size= DOT_SIZE, alpha= DOT_ALPHA)
    p.varea(source=fillsource, x='date', y1=0, y2='value', alpha=0.4)
    p.sizing_mode = 'scale_both'

    p.yaxis.visible = False
    p.ygrid.visible = False
    p.xaxis.formatter = DatetimeTickFormatter(hours = ['%I:%M'])
    p.xaxis.ticker.desired_num_ticks = 12

    return components(p)

def training_data_counts_bar(counts, values):
    """Returns bar chart of training data counts"""

    p = figure(
        x_range= values,
        tools='',
        outline_line_color= None,
        plot_height=300,
        toolbar_location= None,
    )
    p.vbar(x= values, top= counts, width= 0.8, alpha= 0.4)
    p.xgrid.visible = False
    p.yaxis.minor_tick_line_color = None
    p.sizing_mode = 'scale_both'

    return components(p)