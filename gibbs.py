#!/usr/bin/python3

import numpy as np
import argparse
import sys
import random

class Gibbs():

    ''' Gibbs Sampling Algorithm implementation for the given network

    Structure of the input -

    --Character Input
    QueryNode =   location   amenities     neighborhood       children
                  age        price         schools            size

    --Multiple String Inputs

    obsEvidNodes =  Location     location=ugly OR bad OR good              Neighborhood    neighborhood=bad OR good
                    Amenities    amenities=little OR lots                  Children        children=bad OR good
                    Size         size=small OR medium OR large             Schools         schools=bad OR good
                    Price        price=cheap OR ok OR expensive            Age             age=old OR new

                    Command line Inputs - Optional values
                    Giving the values of evidence nodes to be set at the start of the sampling process -e1 -e2 -e3 -e4 -e5 -e6 -e7 -e8

    --Integer Input
    NumUpdates      - Number of Updates to be done                                                            -u
    NumSampleIgnr   - Number of Initial Samples to be ignored before computing the final probability          -d

    -- Input Syntax
    gibbs.py [-h] [-e1 E1] [-e2 E2] [-e3 E3] [-e4 E4] [-e5 E5] [-e6 E6] [-e7 E7] [-e8 E8] [-u U] [-d D]

    '''

    def __init__(self):
        self.numUpdates= 0
        self.numSampleIgnr = 0
        self.QueryNode = 'none'
        self.inpevidenceList = {}
        self.currentScene = {}
        self.locOptions = ['good', 'bad', 'ugly']
        self.sizeOptions = ['small', 'medium', 'large']
        self.childOptions = ['good', 'bad']
        self.amenitiesOptions = ['lots', 'little']
        self.neighOptions = ['bad', 'good']
        self.priceOptions = ['cheap', 'ok', 'expensive']
        self.schooOptions = ['bad', 'good']
        self.ageOptions = ['old', 'new']
        self.allNodes = ['location', 'age', 'schools', 'children', 'neighborhood', 'price', 'size', 'amenities']

    def read_argument(self):

        parser = argparse.ArgumentParser(description='Parse various common line arguments')

        parser.add_argument('QueryNode', type=str,help='A required node to caculate probability for')

        evidences = ['-e1', '-e2', '-e3', '-e4', '-e5', '-e6', '-e7', '-e8']
        for evid in evidences:
            parser.add_argument(evid, type=str, help='An evidence node value - Optional Value')

        parser.add_argument('-u', type=int, help='Number of Updates to be made')
        parser.add_argument('-d', type=int, help='Number of Updates to ignore before computing probability')

        args = parser.parse_args()

        self.numUpdates = args.u
        self.numSampleIgnr = args.d
        self.QueryNode = args.QueryNode

        evidLis = [args.e1, args.e2, args.e3, args.e4, args.e5, args.e6, args.e7, args.e8 ]

        for i in range(0,8):
            # print (evidLis[ic])

            if evidLis[i]!= None:
                ea, eb = str(evidLis[i]).split('=')
            self.inpevidenceList[ea] = eb
            # print (ea, eb)
        print ("Input Evidence List", self.inpevidenceList)

        return self.numUpdates, self.numSampleIgnr, self.QueryNode, self.inpevidenceList

    # Defining the values from the given Conditional Probability Table (CPT) keeping the affecting nodes as conditions
    #
    def CPT_amentiies(self, amen_cond):
        prob_amenities = {}
        prob_amenities = {'lots':0.3, 'little':0.7}
        return  prob_amenities[amen_cond]

    def  CPT_neighbor(self, neigh_cond):
        prob_neighbor = {}
        prob_neighbor = {'bad':0.4, 'good':0.6}
        return prob_neighbor[neigh_cond]

    def CPT_location(self, loc_condition, ame, neighborhood):
        prob_locations = {}
        if ame == 'lots' and neighborhood == 'bad':
            prob_locations = {'good':0.3, 'bad':0.4, 'ugly':0.3}
        elif ame == 'lots' and neighborhood == 'good':
            prob_locations = {'good':0.8, 'bad':0.15, 'ugly':0.05}
        elif ame == 'little' and neighborhood == 'bad':
            prob_locations = {'good':0.2, 'bad':0.4, 'ugly':0.4}
        elif ame == 'little' and neighborhood == 'good':
            prob_locations = {'good':0.5, 'bad':0.35, 'ugly':0.15}
        return  prob_locations[loc_condition]

    def CPT_children(self, child_cond,  neighbs):
        prob_children= {}
        if neighbs == 'good':
            prob_children = {'bad':0.3, 'good':0.7}
        elif neighbs == 'bad':
            prob_children = {'bad': 0.6, 'good': 0.4}
        return prob_children[child_cond]

    def CPT_size(self, size_cond):
        prob_size = {}
        prob_size = {'small':0.33, 'medium':0.34, 'large':0.33}
        return prob_size[size_cond]

    def CPT_schools(self, sch_cond, chidren):
        prob_schools = {}
        if chidren == 'good':
            prob_schools = {'bad':0.8, 'good':0.2}
        elif chidren == 'bad':
            prob_schools = {'bad': 0.7, 'good': 0.3}
        return prob_schools[sch_cond]

    def CPT_age(self, age_cond, loc):
        prob_age = {}
        if loc == 'good':
            prob_age = {'old':0.3, 'new':0.7}
        elif loc == 'bad':
            prob_age = {'old':0.6, 'new':0.4}
        elif loc == 'ugly':
            prob_age = {'old': 0.9, 'new': 0.1}
        return prob_age[age_cond]

    def CPT_price(self, pr_cond, locs, ag, sch, siz):
        prob_price = {}
        if locs == 'good' and ag == 'old' and sch =='bad' and siz == 'small':
            prob_price = {'cheap':0.5 , 'ok':0.4 , 'expensive':0.1}
        elif locs == 'good' and ag == 'old' and sch =='bad' and siz == 'medium':
            prob_price = {'cheap':0.4 , 'ok':0.45 , 'expensive':0.15}
        elif locs == 'good' and ag == 'old' and sch =='bad' and siz == 'large':
            prob_price = {'cheap':0.35 , 'ok':0.45 , 'expensive':0.2}

        elif locs == 'good' and ag == 'old' and sch =='good' and siz == 'small':
            prob_price = {'cheap':0.4 , 'ok':0.3 , 'expensive':0.3}
        elif locs == 'good' and ag == 'old' and sch =='good' and siz == 'medium':
            prob_price = {'cheap':0.35 , 'ok':0.3 , 'expensive':0.35}
        elif locs == 'good' and ag == 'old' and sch =='good' and siz == 'large':
            prob_price = {'cheap':0.3 , 'ok':0.25 , 'expensive':0.45}

        elif locs == 'good' and ag == 'new' and sch =='bad' and siz == 'small':
            prob_price = {'cheap':0.45 , 'ok':0.4 , 'expensive':0.15}
        elif locs == 'good' and ag == 'new' and sch =='bad' and siz == 'medium':
            prob_price = {'cheap':0.4 , 'ok':0.45 , 'expensive':0.15}
        elif locs == 'good' and ag == 'new' and sch =='bad' and siz == 'large':
            prob_price = {'cheap':0.35 , 'ok':0.45 , 'expensive':0.2}

        elif locs == 'good' and ag == 'new' and sch =='good' and siz == 'small':
            prob_price = {'cheap':0.25 , 'ok':0.3 , 'expensive':0.45}
        elif locs == 'good' and ag == 'new' and sch =='good' and siz == 'medium':
            prob_price = {'cheap':0.2 , 'ok':0.25 , 'expensive':0.55}
        elif locs == 'good' and ag == 'new' and sch =='good' and siz == 'large':
            prob_price = {'cheap':0.1 , 'ok':0.2 , 'expensive':0.7}


        if locs == 'bad' and ag == 'old' and sch =='bad' and siz == 'small':
            prob_price = {'cheap':0.7 , 'ok':0.299 , 'expensive':0.001}
        elif locs == 'bad' and ag == 'old' and sch =='bad' and siz == 'medium':
            prob_price = {'cheap':0.65 , 'ok':0.33 , 'expensive':0.02}
        elif locs == 'bad' and ag == 'old' and sch =='bad' and siz == 'large':
            prob_price = {'cheap':0.65 , 'ok':0.32 , 'expensive':0.03}

        elif locs == 'bad' and ag == 'old' and sch =='good' and siz == 'small':
            prob_price = {'cheap':0.55 , 'ok':0.3 , 'expensive':0.15}
        elif locs == 'bad' and ag == 'old' and sch =='good' and siz == 'medium':
            prob_price = {'cheap':0.5 , 'ok':0.35 , 'expensive':0.15}
        elif locs == 'bad' and ag == 'old' and sch =='good' and siz == 'large':
            prob_price = {'cheap':0.45 , 'ok':0.4 , 'expensive':0.15}

        elif locs == 'bad' and ag == 'new' and sch =='bad' and siz == 'small':
            prob_price = {'cheap':0.6 , 'ok':0.35 , 'expensive':0.05}
        elif locs == 'bad' and ag == 'new' and sch =='bad' and siz == 'medium':
            prob_price = {'cheap':0.55 , 'ok':0.35 , 'expensive':0.1}
        elif locs == 'bad' and ag == 'new' and sch =='bad' and siz == 'large':
            prob_price = {'cheap':0.5 , 'ok':0.4 , 'expensive':0.1}

        elif locs == 'bad' and ag == 'new' and sch =='good' and siz == 'small':
            prob_price = {'cheap':0.4 , 'ok':0.4 , 'expensive':0.2}
        elif locs == 'bad' and ag == 'new' and sch =='good' and siz == 'medium':
            prob_price = {'cheap':0.3 , 'ok':0.4 , 'expensive':0.3}
        elif locs == 'bad' and ag == 'new' and sch =='good' and siz == 'large':
            prob_price = {'cheap':0.3 , 'ok':0.3 , 'expensive':0.4}


        if locs == 'ugly' and ag == 'old' and sch =='bad' and siz == 'small':
            prob_price = {'cheap':0.8 , 'ok':0.1999 , 'expensive':0.0001}
        elif locs == 'ugly' and ag == 'old' and sch =='bad' and siz == 'medium':
            prob_price = {'cheap':0.75 , 'ok':0.24 , 'expensive':0.01}
        elif locs == 'ugly' and ag == 'old' and sch =='bad' and siz == 'large':
            prob_price = {'cheap':0.75 , 'ok':0.23 , 'expensive':0.02}

        elif locs == 'ugly' and ag == 'old' and sch =='good' and siz == 'small':
                prob_price = {'cheap':0.65 , 'ok':0.3 , 'expensive':0.05}
        elif locs == 'ugly' and ag == 'old' and sch =='good' and siz == 'medium':
            prob_price = {'cheap':0.6 , 'ok':0.33 , 'expensive':0.07}
        elif locs == 'ugly' and ag == 'old' and sch =='good' and siz == 'large':
            prob_price = {'cheap':0.55 , 'ok':0.37 , 'expensive':0.08}

        elif locs == 'ugly' and ag == 'new' and sch =='bad' and siz == 'small':
            prob_price = {'cheap':0.7 , 'ok':0.27 , 'expensive':0.03}
        elif locs == 'ugly' and ag == 'new' and sch =='bad' and siz == 'medium':
            prob_price = {'cheap':0.64 , 'ok':0.3 , 'expensive':0.06}
        elif locs == 'ugly' and ag == 'new' and sch =='bad' and siz == 'large':
            prob_price = {'cheap':0.61 , 'ok':0.32 , 'expensive':0.07}

        elif locs == 'ugly' and ag == 'new' and sch =='good' and siz == 'small':
            prob_price = {'cheap':0.48 , 'ok':0.42 , 'expensive':0.1}
        elif locs == 'ugly' and ag == 'new' and sch =='good' and siz == 'medium':
            prob_price = {'cheap':0.41 , 'ok':0.39 , 'expensive':0.2}
        elif locs == 'ugly' and ag == 'new' and sch =='good' and siz == 'large':
            prob_price = {'cheap':0.37 , 'ok':0.33 , 'expensive':0.3}

        return prob_price[pr_cond]

    # -------------------Defining CPT ends


    #Defining the Markov Blanket for the various nodes based on the given distribution
    def markov_Blanket(self, node):

        if node == 'amenities':
            list_markov = ['location', 'neighborhood']
        elif node == 'neighborhood':
            list_markov = ['location', 'children', 'amenities']
        elif node == 'children':
            list_markov = ['neighborhood', 'schools']
        elif node == 'location':
            list_markov = ['amenities', 'neighborhood', 'size', 'schools', 'age', 'price']
        elif node == 'age':
            list_markov = ['location', 'schools', 'size', 'price']
        elif node == 'price':
            list_markov = ['location', 'schools', 'size', 'age']
        elif node == 'size':
            list_markov = ['location', 'schools', 'age', 'price']
        elif node == 'schools':
            list_markov = ['location', 'children', 'size', 'age', 'price']

        print ("Markov Blanket for  the node -- ", node," --",  "has the following nodes ", list_markov, "\n")

        return list_markov

    #Defining the function that randomly assigns condition to the node that is input
    #Essentially, it would be the node that is not an evidence node and its value is to be randomly set to start with
    def random_state_gen(self, node):

        if node == 'location':
            self.currentScene['location'] =  self.locOptions[random.randint(0,len(self.locOptions)-1)]
        elif node == 'amenities':
            self.currentScene['amenities'] =  self.amenitiesOptions[random.randint(0,len(self.amenitiesOptions)-1)]
        elif node == 'age':
            self.currentScene['age'] =  self.ageOptions[random.randint(0,len(self.ageOptions)-1)]
        elif node == 'size':
            self.currentScene['size'] =  self.sizeOptions[random.randint(0,len(self.sizeOptions)-1)]
        elif node == 'neighborhood':
            self.currentScene['neighborhood'] =  self.neighOptions[random.randint(0,len(self.neighOptions)-1)]
        elif node == 'price':
            self.currentScene['price'] =  self.priceOptions[random.randint(0,len(self.priceOptions)-1)]
        elif node == 'schools':
            self.currentScene['schools'] =  self.schooOptions[random.randint(0,len(self.schooOptions)-1)]
        elif node == 'children':
            self.currentScene['children'] =  self.childOptions[random.randint(0,len(self.childOptions)-1)]

        return self.currentScene[node]


    #Defining  the function that runs the read_arugment function, validates the input and assigns the random states to non-evidence nodes
    #Finally generates two lists - One with setting of the evidence nodes and Another with the random setting of the non-evidence nodes
    #Also returns the number of updates, number of samples to ignore and query node name
    def nodeValueSetting(self):
        self.numUpdates, self.numSampleIgnr, self.QueryNode, self.inpevidenceList = self.read_argument()
        newdict = {}

        print ("Nodes in the evidence list -- ", list(self.inpevidenceList.keys()))

        if self.QueryNode in list(self.inpevidenceList.keys()):
            print ("\nQuery Node cannot  be an evidence node as well")
            sys.exit("\nChange the inputs \nTerminating the process \nProcess has died - No PID generated - Pretending to be a pro coder")

        print ("---------------")
        for element in self.allNodes:
            # print (element)
            if element not in list(self.inpevidenceList.keys()):
                print ("Node not present in the evidence  -- ", element)
                newdict[element] = self.random_state_gen(element)
                # print (newdict)
        # print ("Non evidence List", newdict)
        # print ("Evidence List", self.inpevidenceList)
        return newdict, self.inpevidenceList, self.numUpdates, self.numSampleIgnr, self.QueryNode


    #Still pondering over this node - I DO NOT REMEMBER IF THIS IS TO BE USED
    def updateNodeValues(self, node):

        marblank = self.markov_Blanket(node)
        return marblank

    #As the description suggests
    #Need to fill in these two functions after all the individual probability distribution functions are made

    # def calcProbSequence(self, nonevidList, inpevidenceList):
    #     '''Calculates probability based on sequential selection of all the non-evidence nodes'''
    #
    # def calcProbRand(self, randNode, nonevidList, inpevidenceList):
    #     ''' Calculates probability based on random node selection for udpate'''


    #Defining the functions for all the nodes to update their probability distribution for random assignment conditioned on the Markov Blanket

    #For Location - Other Nodes to complete

    def probability_location(self, nonevidList, inpevidenceList):

        '''Calculate the probability distribution for location node based on Markov Blanket and then
           normalizing it to get it within the 0-1 range '''

        print ("\nCalculating the probability of the randomly selected --", "location", "--\n")
        _ = self.markov_Blanket('location')

        nonevidList.update(inpevidenceList)
        totalList = nonevidList
        print ("Complete List -- ", totalList)

        prob_locationNewnoNorm, prob_locationNewNormal = {}, {}

        for loc_nodeOption in self.locOptions:
            prob_locationNewnoNorm[loc_nodeOption] = self.CPT_location(loc_nodeOption, totalList['amenities'], totalList['neighborhood'])\
                            *self.CPT_amentiies(totalList['amenities'])*self.CPT_neighbor(totalList['neighborhood'])*\
                            self.CPT_size(totalList['size'])*self.CPT_schools(totalList['schools'], totalList['children'])\
                            *self.CPT_age(totalList['age'], loc_nodeOption)*self.CPT_price(totalList['price'],\
                            loc_nodeOption, totalList['age'], totalList['schools'], totalList['size'])

        summ = sum(list(prob_locationNewnoNorm.values()))
        for key in list(prob_locationNewnoNorm.keys()):
            prob_locationNewNormal[key] = prob_locationNewnoNorm[key]/summ

        print ("Probability distribution of the -- Location -- node without normalization", prob_locationNewnoNorm)
        print ("Probability distribution of the -- Location -- node with normalization", prob_locationNewNormal)

        return 0


#Defining the main function that creates the object for the Class and does some shit - This needs to be structured better

def main():
    gibbs_obj = Gibbs()
    nonevidList, inpevidenceList, numUpdates, numSampleIgnr, QueryNode, = gibbs_obj.nodeValueSetting()

    allValues_noevidList = list(nonevidList.values())
    allValues_noevidList = list(nonevidList.keys())

    allValues_evidList = list(inpevidenceList.values())
    allValues_evidList = list(inpevidenceList.keys())

    print ("Non evidence List", nonevidList)
    print ("Evidence List", inpevidenceList)

    print ("Query Node is -- ", QueryNode)
    print ("Number of updates  -- ", numUpdates)
    print ("Number of initial samples to ignore -- ", numSampleIgnr )
    print ("---------------\n")

    randomNode = allValues_noevidList[random.randint(0, len(allValues_noevidList)-1)]

    if randomNode== 'location':
        _ = gibbs_obj.probability_location(nonevidList, inpevidenceList)

main()

