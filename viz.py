from plane import PlaneModel, PassengerAgent, PatchAgent

from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

colors = [
    'blue', 'cyan', 'orange', 'yellow', 'magenta', 'purple', '#103d3e', '#9fc86c',
    '#b4c2ed', '#31767d', '#31a5fa', '#ba96e0', '#fef3e4', '#6237ac', '#f9cacd', '#1e8123'
]


def agent_portrayal(agent):
    if isinstance(agent, PassengerAgent):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": "grey",
                     "r": 0.9}

        portrayal['Color'] = colors[agent.group - 1]
        if isinstance(agent.group, int) and 1 <= agent.group <= len(colors):
            portrayal['Color'] = colors[agent.group - 1]
        else:
            portrayal['Color'] = "grey"  # Default color for invalid group
        if agent.state == "FINISHED":
            portrayal["Layer"] = 2
        portrayal['text'] = str(agent.unique_id)
        portrayal['text_color'] = 'white'

        portrayal['text'] = agent.unique_id
        portrayal['text_color'] = 'white'

    elif isinstance(agent, PatchAgent):
        portrayal = {"Shape": "rect",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "lightgrey",
                     "w": 1,
                     "h": 1}

        if agent.type == 'CORRIDOR':
            portrayal['Color'] = 'lightgreen'
        elif agent.type == 'SEAT':
            portrayal['Color'] = '#ff6666'

    return portrayal


luggage_vals = ['normal', 0, 1, 2, 3, 4, 5, 6, 7]

class TimeElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return f"Time Elapsed: {model.minute():.1f} minutes"

grid = CanvasGrid(agent_portrayal, 21, 7, 840, 310)
time_element = TimeElement()

method_choice = UserSettableParameter('choice', 'Boarding method', value='Random',
                                                choices=list(PlaneModel.method_types.keys()))
shuffle_choice = UserSettableParameter('checkbox', 'Enable Shuffle', value=True)

bags_choice = UserSettableParameter('choice', 'Luggage Size', value='normal', choices=luggage_vals)

server = ModularServer(PlaneModel,
                       [grid, time_element],
                       "Boarding Simulation",
                       {"method": method_choice, "shuffle_enable": shuffle_choice, 'common_bags': bags_choice})
server.port = 8521 # The default
server.launch()
