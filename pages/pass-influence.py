import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
import os


# import matplotlib.pyplot as plt
# import io
# import base64
# def create_debug_html_component(subG):
#     # Create a matplotlib figure and draw the subgraph
#     plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(subG)  # compute layout for a better visual
#     nx.draw(subG, pos, with_labels=True, node_color='lightblue', 
#             edge_color='gray', node_size=500, font_size=10)
    
#     # Save the figure to a bytes buffer in PNG format
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     plt.close()
#     buf.seek(0)
    
#     # Encode the image in base64 to embed in HTML
#     img_base64 = base64.b64encode(buf.read()).decode("utf-8")
#     html = f'<img src="data:image/png;base64,{img_base64}" alt="Debug Graph" style="display:block;margin-left:auto;margin-right:auto;">'
#     return html


def add_player_node(G, name, team, player_id, jersey_num):
    """
    Add a player node with the real player ID if it doesn't exist.
    Node label example: "CHI_8474141".
    """
    node_key = f"{player_id}"
    if not G.has_node(node_key):
        G.add_node(name,
                   player_id = node_key,
                   team=team, 
                   jersey_num=jersey_num, 
                   type='player')
    return name

def add_shot_node(G, shot_id, is_goal, game_id):
    """
    Add a shot node with an is_goal flag and game_id attribute.
    """
    G.add_node(shot_id, 
               type='shot', 
               is_goal=is_goal, 
               game_id=game_id)
    return shot_id

def process_data_into_graph(df):
    G = nx.DiGraph()

    # Dictionary to track how often one player passes to another
    passing_counts = {}

    # Process each shot attempt in the DataFrame
    for idx, row in df.iterrows():
        # Build a unique shot ID for the node (e.g., shot_0, shot_1, etc.)
        shot_id = f"shot_{idx}"
        
        # Flag if this shot was a goal
        is_goal = True if row['G?'] == 'y' else False
        
        # Capture the game ID from your dataset
        game_id = row['Game ID']
        
        # Create the shot node in the graph
        add_shot_node(G, shot_id, is_goal, game_id) 

        # Add shooter node and connection to shot
        shooter_id = add_player_node(G, row['ShooterName'], row['Team'], row['ShooterPlayerId'], row['Shooter'])
        G.add_edge(shooter_id, shot_id, relationship_type='MAKES_SHOT')

        # Process passing sequences
        passers = []
        for assist in ['A3', 'A2', 'A1']:
            if pd.notna(row[assist]):
                passer_id = add_player_node(G, row[assist+'Name'], row['Team'], row[assist+'PlayerId'], row[assist])
                passers.append(passer_id)

        # Add the passing relationships
        if passers:
            # include shooter in the passing sequence
            passers.append(shooter_id)

            # Create edges for consecutive passes
            for i in range(len(passers)-1):
                passer = passers[i]
                receiver = passers[i+1]

                #update passing counts
                pass_key = (passer, receiver)
                passing_counts[pass_key] = passing_counts.get(pass_key, 0) + 1
        
        #print(passing_counts.items())
        # Add weighted passing relationships
        for (passer, receiver), weight in passing_counts.items():
            G.add_edge(passer, receiver, relationship_type='PASSES_TO', weight=weight)

    return G

def create_player_subgraph(G, team):
    # select only player nodes with the specified team attribute
    player_nodes = [
        n for n, attr in G.nodes(data=True)
        if attr.get('type') == 'player' and attr.get('team') == team 
    ]

    # create a subgraph containing only these nodes
    subG = G.subgraph(player_nodes).copy()

    # remove any edge that is not a PASSES_TO relationship
    for u, v, data in list(subG.edges(data=True)):
        if data.get('relationship_type') != 'PASSES_TO':
            subG.remove_edge(u, v)

    print(subG.nodes)

    return subG



def create_team_subgraph(G, team):
    player_nodes = [
        n for n, attr in G.nodes(data=True)
        if attr.get('team') == team 
    ]
     # create a subgraph containing only these nodes
    subG = G.subgraph(player_nodes).copy() 
    return subG


def filter_down_and_create_assist_network(G, team, min_weight=1):
    # another way to subgraph but testing for the slider
    subG = G.copy()
    print(subG.nodes)
    nodes_to_remove = list(set([n for n, attr in subG.nodes(data=True) if attr['type'] == 'shot'] + [n for n, attr in subG.nodes(data=True) if attr['Team'] != team]))
    subG.remove_nodes_from(nodes_to_remove)

    edges_to_remove = list(set([e for e in subG.edges() if 
                      subG.edges[e].get('relationship_type') != 'PASSES_TO'] + [e for e in subG.edges() if 
                      subG.edges[e].get('width',subG.edges[e]['weight'] ) != 'PASSES_TO']))
    
    subG.remove_edges_from(edges_to_remove)

    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#000000')
    net.from_nx(subG)

    net.set_options("""
    const options = {
        "nodes": {
            "font": {
                "size": 14
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 100
            }
        }
    }
    """)
    
    # Save and display the graph in Streamlit
    try:
        path = '//tmp'
        net.save_graph(f'{path}//pyvis_graph.html')
        html_path = f'{path}//pyvis_graph.html'
    except:
        path = 'html_files'
        net.save_graph(f'{path}//pyvis_graph.html')
        html_path = f'{path}//pyvis_graph.html'

    return html_path



def create_pyvis_assist_network(subG, min_weight=1):
    net = Network(height='600px', width='100%', bgcolor='#FFEFCF', font_color='#1A1815')
    net.from_nx(subG)

    # # customize nodes to display player names nicely
    # for node in net.nodes:
    #     node_data = subG.nodes[node['id']]
    #     player_name = node_data.get('name', 'Unknown')
    #     node['label'] = player_name
    #     # Adjust node size based on the length of the name (tweak multiplier as needed)
    #     node['size'] = 25 + (len(player_name) * 2)
    #     node['title'] = f"Player: {player_name}\nTeam: {node_data.get('team', '')}"
    #     node['color'] = '#1f77b4'

    # Customize PASSES_TO edges
    for edge in net.get_edges():
        #print(type(edge))
        #print(edge)
        #print(min_weight)
        if edge.get('relationship_type') == 'PASSES_TO':
            weight = edge['width']
            if weight >= min_weight:
                edge['weight'] = weight
                edge['title'] = f"Passes: {weight}"
                #edge['width'] = weight * 2
                edge['color'] = '#5D100A'
            else:
                net.edges.remove(edge)
        else:
            net.edges.remove(edge)
    # for node in net.get_nodes():
    #     print(type(node))
    #     print(node)
    #net.show_buttons(filter_=['physics'])
    # Set Pyvis options (unchanged)
    net.set_options("""
    const options = {
        "nodes": {
            "font": {
                "size": 14
            },
            "color":{
                "background":"#BBACA0"
            }
                    
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 100
            }
        }
    }
    """)
    
    # Save and display the graph in Streamlit
    try:
        path = '//tmp'
        net.save_graph(f'{path}//pyvis_graph.html')
        html_path = f'{path}//pyvis_graph.html'
    except:
        path = 'html_files'
        net.save_graph(f'{path}//pyvis_graph.html')
        html_path = f'{path}//pyvis_graph.html'

    return html_path


def calculate_passing_influence(subG, alpha=0.85):
    """
    Calculate passing influence scores using PageRank on the passing network
    
    Parameters:
    - G: Full game network
    - alpha: Damping factor for PageRank (default 0.85)
    
    Returns:
    - Dictionary of player IDs and their influence scores
    - Passing-only subgraph
    """
    
    # extract edge weights for PageRank
    try:
        edge_weights = {(u, v): d['weight'] for u, v, d in subG.edges(data=True)}
        weight='weight'
    except:
        edge_weights = {(u, v): d['width'] for u, v, d in subG.edges(data=True)}
        weight='width'


    # Calculate PageRank with edge weights
    influence_scores = nx.pagerank(subG, alpha=alpha, weight=weight)

    # Sort players by influence score
    ranked_players = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_players




st.set_page_config(page_title="NHL Passing Network", layout="wide")
st.title("NHL Game Passing Network Analysis")

df_interact = pd.read_csv('merged_shots_20116.csv')

# Sidebar controls
st.sidebar.title("Network Controls")
min_passes = st.sidebar.slider("Minimum number of passes (highlights applicable red)", 1, 6, 1)

team_list = ['EDM', 'CHI']
team_list.sort()

# Implement multiselect dropdown menu for option selection
selected_team = st.selectbox('Select team to visualize assist network', team_list)
if not selected_team:
    st.text('Please choose a team to get started')


G = process_data_into_graph(df_interact)
subG = create_player_subgraph(G, selected_team)

# Pyvis interactive
html_path = create_pyvis_assist_network(subG, min_passes)


st.subheader("Interactive Passing Network")
with open(html_path, 'r', encoding='utf-8') as f:
    components.html(f.read(), height=600)


# Passing Influence
ranked_players = calculate_passing_influence(subG, alpha=0.9)

ranked_df = pd.DataFrame({'Player':[], 'PassingInfluence':[]})

for (player, influence_score) in ranked_players:
    df = pd.DataFrame({'Player':[player], 'PassingInfluence':[influence_score]})
    ranked_df = pd.concat([ranked_df, df], ignore_index=True)

st.subheader('Players and Passing Influence')
st.dataframe(ranked_df)
