class Node:
    def __init__(self, parent=None, daughters=[]):
        self.parent = parent
        self.daughters = daughters

def recurse(node, val, goal):
    obj.append(node)
    val += 1
    if val < goal:
        #D1
        node.daughters.append(Node(parent=node))
        recurse(node.daughters[-1], val, goal)

        #D2
        node.daughters.append(Node(parent=node))
        recurse(node.daughters[-1], val, goal)

        #D3
        node.daughters.append(Node(parent=node))
        recurse(node.daughters[-1], val, goal)

        #D4
        node.daughters.append(Node(parent=node))
        recurse(node.daughters[-1], val, goal)

if __name__ == "__main__":
    obj = []
    
    ROOT = Node()
    val = 1
    goal = 10
    recurse(ROOT, val, goal)

    print(obj[-1].daughters)