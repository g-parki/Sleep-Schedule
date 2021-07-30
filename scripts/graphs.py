from datetime import datetime, timedelta

from bokeh.embed import components
from bokeh.layouts import column
from bokeh.models import (BoxAnnotation, DatetimeTickFormatter, Label,
                          LabelSet, PanTool, Range1d, RangeTool, ResetTool,
                          WheelZoomTool)
from bokeh.models.layouts import Column
from bokeh.models.tools import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from bokeh.transform import jitter

from scripts.datahelpers import convert_timezone_np


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


def bedtime_graph(sourceDF, fillsourceDF):
    """Returns script, div of a plot of automatic readings over last 24 hours"""
    
    tools = [PanTool()]
    TOOLTIPS = '<div"><img src= "@photo_url"><p>@file_name</p></div>'
    DOT_SIZE = 10
    DOT_ALPHA = .20

    source = ColumnDataSource(
        data= dict(
            date= convert_timezone_np(sourceDF.index),
            value= 0.5*(sourceDF.apply(lambda row: row['value']=='Baby', axis=1))+0.25,
            photo_url= sourceDF['photo_url'],
            file_name= sourceDF['file_name']
        )
    )

    fillsource = ColumnDataSource(
        data= dict(date= convert_timezone_np(fillsourceDF.index),
            value= 0.5*(fillsourceDF.apply(lambda row: row['value']=='Baby', axis=1))+0.25,
        )
    )

    p = figure(
        x_axis_type='datetime',
        tools= tools,
        toolbar_location= None,
        toolbar_sticky= False,
        x_axis_location="above",
        x_range=Range1d(start=datetime.now() - timedelta(hours=24),
            end= datetime.now(),
            bounds= (
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
        ),
        y_range=Range1d(start=-.07, end=1.07, bounds=(-.07,1.07)),
        plot_height=300,
        outline_line_color= None
    )
    #Main blue dots
    dots = p.circle('date', jitter('value', .05), source= source, size= DOT_SIZE, alpha= DOT_ALPHA, hover_alpha= 0.9)
    dots.glyph.line_width = 5
    dots.glyph.line_alpha = 0
    
    #Vertical line hover tool
    #Based on example https://docs.bokeh.org/en/latest/docs/user_guide/styling.html
    p.add_tools(HoverTool(tooltips= TOOLTIPS, renderers=[dots], mode= 'vline'))
    
    #Bar graph made of upsampled data
    p.varea(source=fillsource, x='date', y1=0.25, y2='value', alpha=0.3)

    #Range tool for scrolling
    select = figure(
        y_range = p.y_range,
        x_range = p.x_range.bounds,
        x_axis_type='datetime',
        y_axis_type=None,
        tools='',
        toolbar_location= None,
        plot_height=75,
        outline_line_color= None,
        title="Scroll",
        title_location = 'above',
    )
    select.title.align = 'right'
    select.title.text_font_size = '10px'
    select.title.text_font_style = 'normal'
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_alpha = 0.5

    select.varea(source=fillsource, x='date', y1=0.25, y2='value', alpha=.2)
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    select.sizing_mode = "stretch_width"
    
    
    p.sizing_mode = "stretch_both"
    p.yaxis.visible = False
    p.ygrid.visible = False
    p.xaxis.formatter = DatetimeTickFormatter(hours = ['%I:%M'])
    p.xaxis.ticker.desired_num_ticks = 12

    return components(column(p, select, sizing_mode="stretch_both"))

def training_data_counts_bar(counts, value_names):
    """Returns bar chart of training data counts"""
    source = ColumnDataSource(data=dict(categories= value_names, counts= counts))
    p = figure(
        x_range= value_names,
        tools='',
        outline_line_color= None,
        plot_height=300,
        toolbar_location= None,
    )
    
    p.vbar(x= 'categories', top= 'counts', width= 0.6, alpha= 0.4, source=source)
    labels = LabelSet(x= 'categories', y= 'counts', text= 'counts', y_offset=1, x_offset=-13, source=source, render_mode= 'css')
    p.add_layout(labels)

    p.xgrid.visible = False
    p.yaxis.minor_tick_line_color = None
    p.sizing_mode = 'stretch_both'

    return components(p)


def model_performance_graph_quads(baby_df, nobaby_df, dims):
    """Returns script, div of a plot of a model's predictions for the entire dataset"""

    #HTML tooltip which references the photo_url in the dataframe
    tools = [PanTool(), WheelZoomTool(maintain_focus= False), ResetTool()]
    TOOLTIPS = '<div><img src= "@photo_url"><p>@file_name</p></div>'

    baby_src = ColumnDataSource(data= baby_df)
    nobaby_src = ColumnDataSource(data= nobaby_df)
    JITTER_RADIUS_X = .97
    JITTER_RADIUS_Y = .47
    DOT_SIZE = 10
    DOT_ALPHA = .1
    BACK_ALPHA = .05

    p = figure(
        tooltips= TOOLTIPS,
        tools=tools,
        toolbar_location=None,
        toolbar_sticky=False,
        active_scroll= tools[1],
        x_axis_location="above",
        x_range=Range1d(start=dims.get('x_start'), end=dims.get('x_end'), bounds=(dims.get('x_start'),dims.get('x_end'))),
        y_range=Range1d(start=dims.get('y_start'), end=dims.get('y_end'), bounds=(dims.get('y_start'),dims.get('y_end'))),
        plot_height=200,
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
    red_dots = p.circle(
        jitter('NoBabyLikeliness', JITTER_RADIUS_X),
        jitter('y', JITTER_RADIUS_Y),
        source= nobaby_src,
        size=DOT_SIZE,
        color='red', alpha=DOT_ALPHA,
        #legend_label="Photos without baby  ",
        #legend=False
    )
    blue_dots = p.circle(
        jitter('BabyLikeliness', JITTER_RADIUS_X),
        jitter('y', JITTER_RADIUS_Y),
        source= baby_src,
        size=DOT_SIZE,
        color='blue', alpha=DOT_ALPHA,
        #legend_label="Photos with baby   ",
        #legend=False
    )

    red_dots.glyph.line_width = 5
    red_dots.glyph.line_alpha = 0
    blue_dots.glyph.line_width = 5
    blue_dots.glyph.line_alpha = 0

    p.line([-1.5, 1.5], [.5, .5], line_width=1, color='dimgray')#horizontal
    p.line([0, 0], [-1.5, 1.5], line_width=1, color='dimgray')#vertical

    p.sizing_mode = 'stretch_width'

    #Axis config
    p.yaxis.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.xgrid.visible = False
    p.xaxis.axis_label_text_font_size = '8pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.major_label_text_font_size = '0pt'  #turn off x-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt' #turn off y-axid tick labels

    p.min_border_left = 0
    p.min_border_right = 0
    p.min_border_top = 0
    p.min_border_bottom = 0

    #Legend config
    # p.legend.location = None
    # p.legend.click_policy="hide"
    # p.legend.label_text_font_size = '8pt'
    # p.legend.label_text_font_style = 'normal'
    # p.legend.orientation='vertical'
    # p.legend.glyph_height= 20
    # p.legend.background_fill_alpha = 0.7
    # p.legend.border_line_width = 1


    return components(p)