import numpy as np
import pandas as pd
import seaborn as sns

def shorten_relationship_type(str):
    ### Given a biological_relationship_to_proband string, return a shortened version
    ### for cousins and half siblings

    # if string contains 'half sibling with a shared'
    if 'half sibling with a shared' in str.lower():
        return 'Half Sibling'
    if 'aternal cousin' in str.lower():
        return 'First Cousin'
    else:
        return str

def map_simplified_tree_nodes(nodes_array, nodes_map):
    ### The nth value of nodes_map is the node id of the nth node in the original tree sequence
    ### in the simplified tree sequence. If this node doesn't exist in the simplified ts, the value
    ### is -1. This function inverts nodes_map: given an np.array of node id's in the simplified ts, it returns
    ### an np.array of node id's in the original ts.

    nodes_map_inv = np.zeros(len(nodes_map), dtype=int)
    nodes_map_inv[nodes_map] = np.arange(len(nodes_map))
    return nodes_map_inv[nodes_array]    

def make_span_df(ts, n_per_sample, metadata_cols=['participant_id', 'biological_relationship_to_proband', 
                                       'rare_diseases_family_sk']):
    ### Given a tree sequence, determine an n x n array in which the (i,j)th entry is the 
    ### span of the genome in which the ith and jth samples are siblings in a subtree. 
    ### Melt the dataframe and add on metadata given by the metadata cols.

    sample_span = np.zeros((ts.num_samples, ts.num_samples))

    for tree in ts.simplify().trees(sample_lists=False):
        #print(f"Visiting tree {tree.index}")
        if tree.has_single_root:
            for u in ts.samples():
                #print(f"  Checking neighbours for sample {u} under parent {tree.parent(u)}")
                for nearest in tree.samples(tree.parent(u)):
                    if nearest != u:
                        #print(f"    {nearest} is a neighbour of {u}")
                        #print(f"    Adding {tree.span} to the sample span between {u} and {nearest}")
                        sample_span[u, nearest] += tree.span
    sample_span = sample_span / ts.sequence_length
    samples = [str(s) for s in ts.samples()]
    
    #Construct the dataframe
    span_df = pd.DataFrame(sample_span.T, columns = samples)
    span_df['sample'] = samples
    
    #Melt the dataframe for plotting
    span_melted = span_df.melt(id_vars=['sample'])
    span_melted.set_index('sample',drop=True, inplace=True)
    df = span_melted.groupby('variable')['value'].nlargest(n_per_sample).reset_index()
    df.columns = ['Sample','Neighbour','Value']

    #For each sample node in ts_small, extract the corresponding biological_relationship_to_proband
    #and rare_diseases_family_sk metadata from the associated individual and store it in a dataframe
    #with the node id as the index
    node_metadata = pd.DataFrame(index=ts.samples().astype(int), columns=metadata_cols)
    for node_id in ts.samples():
        ind_id = ts.node(node_id).individual
        ind_metadata = ts.individual(ind_id).metadata
        node_metadata.loc[node_id] = pd.Series({k: v for k, v in ind_metadata.items() if k in metadata_cols})

    #Apply shorten metadata function over biological_relationship_to_proband column
    node_metadata['biological_relationship_to_proband'] = node_metadata['biological_relationship_to_proband'].apply(shorten_relationship_type) 

    # get the unique values in the rare_diseases_family_sk column
    unique_values = node_metadata['rare_diseases_family_sk'].unique()
    # create a dictionary that maps each unique value to its corresponding "Family n" string
    family_dict = {value: f"Family {i+1}" for i, value in enumerate(unique_values)}
    # replace the values in the rare_diseases_family_sk column using the dictionary
    node_metadata['Family'] = node_metadata['rare_diseases_family_sk'].map(family_dict)
    
    #Left join topn with node_metadata on Sample
    df['Sample'] = df['Sample'].astype(int)
    df = df.merge(node_metadata, how='left', left_on='Sample', right_index=True)

    df['Sample'] = df['Sample'].astype(str)
    df['Neighbour'] = df['Neighbour'].astype(str)
    df['participant_id'] = df['participant_id'].astype(str)
    df.sort_values(by='Sample', inplace=True, key=lambda x : x.astype(int))

    return df

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import rgb2hex

def get_colors(n, colormap):
    ### Return a list of n hex colors from a given matplotlib colormap
    ### (https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
    
    cmap = ListedColormap(plt.get_cmap(colormap).colors)
    colors = cmap.colors[:n]
    hex_colors = [rgb2hex(color) for color in colors]
    
    return hex_colors

def format_node_label(node_id, df):
    ### Given a dataframe of nodes and metadata and a node id
    ### Return a formatted string for the node label

    # Get the first row of the dataframe where the Sample is node_id

    row = df[df['Sample'] == node_id].drop_duplicates(subset=['Sample'], keep='first')   
    family_id = row['Family'].iloc[0]
    bio_rel = row['biological_relationship_to_proband'].iloc[0]
    # Return the formatted string
    return f"Node {node_id} - {family_id}'s {bio_rel}"

def get_id_of_sample(sample, df):
    ### Return the participant_id of a given sample from a given dataframe (df).
    return df.loc[df['Sample'] == str(sample), 'participant_id'].values[0]

def plot_x_axis_labels(column_name, bracket_height, df, ax, participantQ, arrow_style, palette='tab20'):
    ### Plot x-axis labels for a given column_name with lines/brackets at a given height
    ### (bracket_height) on a given axis (ax) with style (arrow_style).
    ### The labels are taken from the given dataframe (df).
    ### If participantQ is True, the labels are the relationship_to_proband values, otherwise they are
    ### the values of the given column_name.

    # if participantQ is True, we need to get a color palette for the participant labels
    if participantQ:
        # get colors for all samples from palette
        colors = get_colors(len(df['Sample'].unique()), palette)
        # get every other color from list starting with 0
        colors = colors[::2]

    # get the unique values for the given column_name
    values = df[column_name].unique()
    # loop over the unique values
    for i, value in enumerate(values):
        # get the Samples belonging to the current value
        samples = df.loc[df[column_name] == value, 'Sample'].values
        # get the minimum and maximum x-axis values for the current value
        x_min, x_max = ax.get_xlim()
        x_values = [i for i, sample in enumerate(ax.get_xticks()) if str(sample) in samples]
        x_min = min(x_values)-0.5
        x_max = max(x_values)+0.5

            
        # add the value annotation
        x_pos = x_min + (x_max - x_min) / 2
        if participantQ:
            # get the relationship_to_proband value for the participant
            relationship = df.loc[df['participant_id'] == value, 'biological_relationship_to_proband'].values[0]
            ax.annotate(relationship, xy=(x_pos, bracket_height - 0.05), ha='center', va='center',
                            annotation_clip=False, color=colors[i])
            # draw the bracket
            ax.annotate('', xytext=(x_min, bracket_height), xy=(x_max, bracket_height),
                        arrowprops=dict(arrowstyle=arrow_style, color=colors[i]), annotation_clip=False)
        else:
            ax.annotate(value, xy=(x_pos, bracket_height - 0.05), ha='center', va='center',
                            annotation_clip=False)
            # draw the bracket
            ax.annotate('', xytext=(x_min, bracket_height), xy=(x_max, bracket_height),
                        arrowprops=dict(arrowstyle=arrow_style, color='black'), annotation_clip=False)

def plot_span_df(df, ts, palette = 'tab20', normaliseQ = False, fig_width=10):
    ### Plot a span dataframe (df) with a given tree sequence (ts) and color palette (palette).
    ### If normaliseQ is True, the span values are normalised such that the bars are of equal height.

    #neighbours_list ordered for plotting
    neighbours = df['Neighbour'].drop_duplicates().to_list()
    neighbours.sort(key = int)
    
    #using histplot because it can create stacked bar charts nicely
    if normaliseQ:
        ax = sns.histplot(df, x='Sample', hue='Neighbour', weights='Norm_Value',
                      multiple='stack', shrink=0.8, hue_order = neighbours, palette=palette)
    else:
        ax = sns.histplot(df, x='Sample', hue='Neighbour', weights='Value',
                      multiple='stack', shrink=0.8, hue_order = neighbours, palette=palette)

    #neighbours_list ordered for plotting
    neighbours = df['Neighbour'].drop_duplicates().to_list()
    neighbours.sort(key = int)
    #apply format_node_label to each element of the neighbours list and column
    neighbours = [format_node_label(x, df) for x in neighbours]
    df['Neighbour'] = df['Neighbour'].apply(lambda x: format_node_label(x, df))
        
    #using histplot because it can create stacked bar charts nicely
    ax = sns.histplot(df, x='Sample', hue='Neighbour', weights='Value', bins=10,
                        multiple='stack', shrink=0.8, hue_order = neighbours, palette='tab20')
    ax.set(xlabel=None, ylabel='Span')
    #add labels two layers of x axis labels
    plot_x_axis_labels(column_name='participant_id', bracket_height=-0.12, df=df, ax=ax, participantQ=True,
                    arrow_style='-', palette=palette)

    plot_x_axis_labels(column_name='Family', bracket_height=-0.25, df=df, ax=ax,
                        participantQ=False, arrow_style='|-|')

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # Set figure size to 20 x 10
    plt.rcParams['figure.figsize'] = [fig_width, fig_width/1.618]
    plt.show()

def plot_span_ts(ts, n_per_sample=3, palette='tab20', normaliseQ=False, fig_width=10):
    ### Plot a span dataframe (df) with a given tree sequence (ts) and color palette (palette).
    df = make_span_df(ts, n_per_sample)
    plot_span_df(df, ts, palette, normaliseQ, fig_width)

def get_other_samples(sample_sets, ts):
    ### Given a list of sample lists (sample_sets), generate a list of 
    ### all other samples in the ts and output the results in a list format
    ### for the topology.counter method

    #assert that every element of sample_sets is a list
    assert all(isinstance(x, list) for x in sample_sets)
    # get all samples from ts
    all_samples = ts.samples()
    # create a list of all samples not in sample_list
    other_samples = [sample for sample in all_samples if sample not in [item for sublist in sample_sets for item in sublist]]
    # create a list of lists of all other samples
    sample_sets.append(other_samples)
    return sample_sets

def count_topologies(ts, sample_sets):
    ### Loop through the trees in the ts, recording the count of every
    ### possible topology in each tree into a dataframe
    
    sample_sets_all = get_other_samples(sample_sets, ts)
    all_ranks = [ts.rank() for ts in tskit.all_trees(num_leaves = len(sample_sets_all))]
    all_nodes = tuple(range(len(sample_sets_all))) #to extract topologies including all nodes
    counts_df = pd.DataFrame(np.zeros((ts.num_trees, len(all_ranks))), columns=all_ranks).astype(int)
    
    for tree, tc in enumerate(ts.count_topologies(sample_sets=sample_sets_all)):
        for rank, count in tc.topologies[all_nodes].items():
            counts_df[rank][tree] = count
    
    counts_df['LeftInterval'] = [t.interval[0] for t in ts.trees()]
    counts_df['RightInterval'] = [t.interval[1] for t in ts.trees()]
    counts_df['Width'] = counts_df['RightInterval'] - counts_df['LeftInterval']
    #counts_df['TreeIndex'] = range(ts.num_samples+1)
    return counts_df

def plot_counts(counts_df, output_path=None, n_ranks=3, color_map='tab10', 
                plot_dim=(800,600)):
    ### Plot the top n most common topologies in the ts
    ### and return the total counts of each topology with
    ### associated colors for tree legend plot
    width_px = plot_dim[0]
    height_px = plot_dim[1]
    
    excl_columns = ['LeftInterval','RightInterval','Width'] #we just want the counts
    counts = counts_df.drop(excl_columns, axis=1)
    totals = counts.sum(axis=0).reset_index() #count across all trees
    totals.columns = ["Rank", "Count"]
    totals.sort_values(by="Count", ascending=False, inplace=True)
    topn_totals = totals.head(n_ranks).reset_index() #get the top n most common topologies
    topn_counts = counts[topn_totals['Rank']]

    x = counts_df['LeftInterval'].to_numpy()
    rightmost = counts_df['RightInterval'].to_list()[-1]
    x = np.append(x, rightmost)

    # For some reason, if there are n samples then stackplot needs n+1 points on the x axis;
    #  however, it still requires n+1 height values for the n stacks. 
    # Hence, we add a dummy row to the heights which is never plotted.
    heights = topn_counts.astype(int).to_numpy()
    last = [heights[-1]]
    heights = np.vstack([heights, last])
    
    # Create the stacked bar chart using stackplot
    hex_colors = get_colors(n_ranks, color_map)
    fig, ax = plt.subplots()
    ax.stackplot(x/1000, heights.T, step='post', colors=hex_colors)

    # Set the y-axis label and limit
    ax.set_ylabel('Count')
    ax.set_xlabel('Genomic position (kBP)')
    px = 1/plt.rcParams['figure.dpi']
    fig.set_size_inches(width_px*px, height_px*px)

    # Save the plot and show it
    if output_path is not None:
        plt.savefig(output_path+'.svg', bbox_inches='tight')
    plt.show()
    #Return the top n most common topologies and their counts for further analysis
    #hex_colors.reverse()
    return topn_totals, hex_colors

from IPython.display import SVG, display

def annotate_sample_nodes(sample_sets):
    ### Given a list of sample sets, return a dictionary of node IDs and labels
    ### for an unranked tree which maps the sample nodes (0,1..) on the tree to the
    ### sample_sets, with the final node being mapped to 'x'.
    ### e.g. if sample_list = [[0],[2],[3,4]] the dictionary will be
    ### {0: 0, 1: 2, 2: '[3,4]', 3: 'x'}
    ### where 0,1,2,3 are the node IDs on the tree
    sample_dict = {}
    for i, sample_set in enumerate(sample_sets):
        if len(sample_set) == 1:
            sample_dict[i] = sample_set[0]
        else:
            sample_dict[i] = str(sample_set)
    sample_dict[len(sample_sets)] = 'x'
    return sample_dict

def plot_topn_trees(topn_totals, hex_colors, sample_sets, output_path=None,
                    plot_dim=(200,120), scaling_factor=(1.1,4), lab_font_size=12):
    ### Plot the top n most common topologies in the ts with edge colors
    ### corresponding to the colors in the stacked bar chart
    style_str = (f".x-axis .lab {{transform: translateY(8px); font-size: {lab_font_size}px}}")
    n_trees = len(topn_totals)
    final_width = plot_dim[0]
    final_height = plot_dim[1]
    tree_width = round(plot_dim[0]/n_trees)
    x_scale = scaling_factor[0]
    y_scale = scaling_factor[1]
    svg_ls = []

    for index, row in topn_totals.iterrows():
        style_str += f'#tree{index} .edge {{stroke: {hex_colors[index]}; stroke-width: 2px}}'
        style_str += f'#tree{index} .tree {{transform: translateX({index*tree_width}px)}}'
        rank_str = f"Rank: ({row['Rank'].shape}, {row['Rank'].label}); n={row['Count']}"  
        t = tskit.Tree.unrank(num_leaves=len(sample_sets)+1, rank=row['Rank'])
        svg = t.draw_svg(node_labels=annotate_sample_nodes(sample_sets), 
                         root_svg_attributes={'id': f'tree{index}',
                                              'viewBox': f'0 0 {final_width} {final_height}',
                                              'preserveAspectRatio': 'xMinYMin meet'},
                        order="tree", size=(tree_width, final_height), style=style_str, x_axis=False,
                        x_label=rank_str)
        svg_ls.append(svg)
        #display(SVG(svg))
    svg = f'<svg baseProfile="full" version="1.2" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" preserveAspectRatio="xMinYMin meet" viewBox="0 0 {final_width/x_scale} {final_height/y_scale}">'# height="{final_height}" width="{final_width}" >' 
    #print(svg)
    svg += ''.join(svg_ls)
    svg += '</svg>'
    
    display(SVG(svg))
    if output_path is not None:
        with open(output_path + '.svg', 'w') as f:
            f.write(svg)

    return svg


def plot_topology_from_df(counts_df, sample_sets, n_ranks, stack_dim, tree_dim, stack_path=None, trees_path=None, 
                   scaling_factor=(1.1,4), color_map='tab10', lab_font_size=12):
    ### Given a dataframe of rank counts, plot the top n most common topologies 
    ### in a stacked bar chart with a legend showing what the ranks look like

    #print("Plotting stacked bar chart...")
    topn_totals, hex_colors = plot_counts(counts_df, stack_path, n_ranks=n_ranks, color_map=color_map,
                                            plot_dim=stack_dim)
    #print("Plotting the top n most common topologies...")
    plot_topn_trees(topn_totals, hex_colors, sample_sets, output_path=trees_path,
                                plot_dim=tree_dim, scaling_factor=scaling_factor, lab_font_size=lab_font_size)
    
