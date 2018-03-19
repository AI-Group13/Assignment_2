#!/usr/bin/python3

import numpy as np
import argparse
import sys
import random
from collections import Counter
import time
from matplotlib import pyplot as plt

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
                    Evidence Nodes can be entered without a prefix
                    

    --Integer Input
    NumUpdates      - Number of Updates to be done                                                            -u
    NumSampleIgnr   - Number of Initial Samples to be ignored before computing the final probability          -d
    [Prefix Required for no. of updates and ignored no. of samples]

    -- Input Syntax
    gibbs.py [-h] [E1] [E2] [E3] [E4] [E5] [E6] [E7] [E8] [-u U] [-d D]

    -- Example Input command
    python3 gibbs.py location neighborhood=good amenities=lots -u 10000 -d 500
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
        
        self.locationStates = {}
        self.neighborhoodStates = {}
        self.amenitiesStates = {}
        self.childrenStates = {}
        self.sizeStates = {}
        self.schoolsStates = {}
        self.ageStates = {}
        self.priceStates = {}
        self.evidLis = []
        
    def read_argument(self):

        parser = argparse.ArgumentParser(description='Parse various common line arguments')

        parser.add_argument('QueryNode', type=str,help='A required node to caculate probability for')
        parser.add_argument('EvidenceNodes',  nargs='*', type=str, help='An evidence node value - Optional Value')
        parser.add_argument('-u', type=int, help='Number of Updates to be made')
        parser.add_argument('-d', type=int, help='Number of Updates to ignore before computing probability', default = 0)

        args = parser.parse_args()

        self.numUpdates = args.u
        self.numSampleIgnr = args.d
        self.QueryNode = args.QueryNode

        for nodeValue in args.EvidenceNodes:
            self.evidLis.append(nodeValue)

        for nodes in self.evidLis:
            ea, eb = nodes.split('=')
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

#        print ("Markov Blanket for  the node -- ", node," --",  "has the following nodes ", list_markov, "\n")

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

#        print ("\nCalculating the probability of the randomly selected --", "location", "--\n")
        _ = self.markov_Blanket('location')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
        #print ("Complete List -- ", totalList)

        prob_locationNewnoNorm, prob_locationNewNormal = {}, {}
        for loc_nodeOption in self.locOptions:
            prob_locationNewnoNorm[loc_nodeOption] = self.CPT_location(loc_nodeOption, totalList['amenities'], totalList['neighborhood'])\
                            *self.CPT_amentiies(totalList['amenities'])*self.CPT_neighbor(totalList['neighborhood'])*\
                            self.CPT_size(totalList['size'])*self.CPT_schools(totalList['schools'], totalList['children'])\
                            *self.CPT_age(totalList['age'], loc_nodeOption)*self.CPT_price(totalList['price'],\
                            loc_nodeOption, totalList['age'], totalList['schools'], totalList['size'])
        
        #p(location|amenities,neighborhood)*p()
        summ = sum(list(prob_locationNewnoNorm.values()))
        for key in list(prob_locationNewnoNorm.keys()):
            prob_locationNewNormal[key] = prob_locationNewnoNorm[key]/summ

#        print ("Probability distribution of the -- Location -- node without normalization", prob_locationNewnoNorm)
#        print ("Probability distribution of the -- Location -- node with normalization", prob_locationNewNormal)
        
        Update_value = np.random.choice(['good','bad','ugly'],p=[prob_locationNewNormal['good'],prob_locationNewNormal['bad'],prob_locationNewNormal['ugly']])
        return Update_value


    def probability_amenities(self, nonevidList, inpevidenceList):

        '''Calculate the probability distribution for amenities node based on Markov Blanket and then
           normalizing it to get it within the 0-1 range '''
        
#        print ("\nCalculating the probability of the randomly selected --", "amenities", "--\n")
        _ = self.markov_Blanket('amenities')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
#        print(totalList['amenities'])
        #print ("Complete List -- ", totalList)

        prob_amenitiesNewnoNorm, prob_amenitiesNewNormal = {}, {}
        for amn_nodeOption in self.amenitiesOptions:
            prob_amenitiesNewnoNorm[amn_nodeOption] = self.CPT_location(totalList['location'], amn_nodeOption, totalList['neighborhood'])\
                            *self.CPT_amentiies(amn_nodeOption)*self.CPT_neighbor(totalList['neighborhood'])
        
        #p(location|amenities,neighborhood)*p()
        summ = sum(list(prob_amenitiesNewnoNorm.values()))
        for key in list(prob_amenitiesNewnoNorm.keys()):
            prob_amenitiesNewNormal[key] = prob_amenitiesNewnoNorm[key]/summ

#        print ("Probability distribution of the -- amenities -- node without normalization", prob_amenitiesNewnoNorm)
#        print ("Probability distribution of the -- amenities -- node with normalization", prob_amenitiesNewNormal)
        
        Update_value = np.random.choice(['lots','little'],p=[prob_amenitiesNewNormal['lots'],prob_amenitiesNewNormal['little']])
        return Update_value

    def probability_neighborhood(self, nonevidList, inpevidenceList):

        '''Calculate the probability distribution for neighborhood node based on Markov Blanket and then
           normalizing it to get it within the 0-1 range '''
        
#        print ("\nCalculating the probability of the randomly selected --", "neighborhood", "--\n")
        _ = self.markov_Blanket('neighborhood')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
#        print(totalList['neighborhood'])
        #print ("Complete List -- ", totalList)

        prob_neighborhoodNewnoNorm, prob_neighborhoodNewNormal = {}, {}
        for neigh_nodeOption in self.neighOptions:
            prob_neighborhoodNewnoNorm[neigh_nodeOption] = self.CPT_neighbor(neigh_nodeOption)*self.CPT_children(totalList['children'],neigh_nodeOption)\
                                                            *self.CPT_location(totalList['location'],totalList['amenities'],neigh_nodeOption)\
                                                            *self.CPT_amentiies(totalList['amenities'])
                                                            
        #p(location|neighborhood,neighborhood)*p()
        summ = sum(list(prob_neighborhoodNewnoNorm.values()))
        for key in list(prob_neighborhoodNewnoNorm.keys()):
            prob_neighborhoodNewNormal[key] = prob_neighborhoodNewnoNorm[key]/summ

#        print ("Probability distribution of the -- neighborhood -- node without normalization", prob_neighborhoodNewnoNorm)
#        print ("Probability distribution of the -- neighborhood -- node with normalization", prob_neighborhoodNewNormal)
        
        Update_value = np.random.choice(['bad','good'],p=[prob_neighborhoodNewNormal['bad'],prob_neighborhoodNewNormal['good']])
        return Update_value

    def probability_size(self, nonevidList, inpevidenceList):

        '''Calculate the probability dis\ntribution for size node based on Markov Blanket and then
           normalizing it to get it within the 0-1 range '''
        
#        print ("\nCalculating the probability of the randomly selected --", "size", "--\n")
        _ = self.markov_Blanket('size')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
        #print ("Complete List -- ", totalList)

        prob_sizeNewnoNorm, prob_sizeNewNormal = {}, {}
        for size_nodeOption in self.sizeOptions:
            prob_sizeNewnoNorm[size_nodeOption] = self.CPT_size(size_nodeOption)*self.CPT_price(totalList['price'], totalList['location'], totalList['age']\
                              ,totalList['schools'],size_nodeOption)*self.CPT_age(totalList['age'],totalList['location'])\
                              *self.CPT_location(totalList['location'],totalList['amenities'],totalList['neighborhood'])*\
                              self.CPT_schools(totalList['schools'],totalList['children'])
        
        #p(location|size,size)*p()
        summ = sum(list(prob_sizeNewnoNorm.values()))
        for key in list(prob_sizeNewnoNorm.keys()):
            prob_sizeNewNormal[key] = prob_sizeNewnoNorm[key]/summ

#        print ("Probability distribution of the -- size -- node without normalization", prob_sizeNewnoNorm)
#        print ("Probability distribution of the -- size -- node with normalization", prob_sizeNewNormal)
        
        Update_value = np.random.choice(['small','medium','large'],p=[prob_sizeNewNormal['small'],prob_sizeNewNormal['medium'],prob_sizeNewNormal['large']])
        return Update_value
    
    def probability_children(self, nonevidList, inpevidenceList):

        '''Calculate the probability distribution for children node based on Markov Blanket and then
           normalizing it to get it within the 0-1 range '''
        
#        print ("\nCalculating the probability of the randomly selected --", "children", "--\n")
        _ = self.markov_Blanket('children')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
#        print(totalList['children'])
        #print ("Complete List -- ", totalList)

        prob_childrenNewnoNorm, prob_childrenNewNormal = {}, {}
        for children_nodeOption in self.childOptions:
            prob_childrenNewnoNorm[children_nodeOption] = self.CPT_children(children_nodeOption,totalList['neighborhood'])*self.CPT_neighbor(totalList['neighborhood'])\
                                  *self.CPT_schools(totalList['schools'],children_nodeOption)
            
        #p(location|children,children)*p()
        summ = sum(list(prob_childrenNewnoNorm.values()))
        for key in list(prob_childrenNewnoNorm.keys()):
            prob_childrenNewNormal[key] = prob_childrenNewnoNorm[key]/summ
#
#        print ("Probability distribution of the -- children -- node without normalization", prob_childrenNewnoNorm)
#        print ("Probability distribution of the -- children -- node with normalization", prob_childrenNewNormal)
        
        Update_value = np.random.choice(['bad','good'],p=[prob_childrenNewNormal['bad'],prob_childrenNewNormal['good']])
        return Update_value
    
    def probability_schools(self, nonevidList, inpevidenceList):

        '''Calculate the probability distribution for schools node based on Markov Blanket and then
           normalizing it to get it with        print ("Probability distribution of the -- schools -- node without normalization", prob_schoolsNewnoNorm)
        print ("Probability distribution of the -- schools -- node with normalization", prob_schoolsNewNormal)in the 0-1 range '''
        
#        print ("\nCalculating the probability of the randomly selected --", "schools", "--\n")
        _ = self.markov_Blanket('schools')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
#        print(totalList['schools'])
        #print ("Complete List -- ", totalList)

        prob_schoolsNewnoNorm, prob_schoolsNewNormal = {}, {}
        for schools_nodeOption in self.schooOptions:
            prob_schoolsNewnoNorm[schools_nodeOption] = self.CPT_schools(schools_nodeOption,totalList['children'])*\
                                                        self.CPT_children(totalList['children'],totalList['neighborhood'])*\
                                                        self.CPT_price(totalList['price'], totalList['location'], totalList['age'],schools_nodeOption,totalList['size'])\
                                                        *self.CPT_age(totalList['age'], totalList['location'])* self.CPT_location(totalList['location'], totalList['amenities'], totalList['neighborhood'])\
                                                        *self.CPT_size(totalList['size'])
        
            
        #p(location|schools,schools)*p()
        summ = sum(list(prob_schoolsNewnoNorm.values()))
        for key in list(prob_schoolsNewnoNorm.keys()):
            prob_schoolsNewNormal[key] = prob_schoolsNewnoNorm[key]/summ

#        print ("Probability distribution of the -- schools -- node without normalization", prob_schoolsNewnoNorm)
#        print ("Probability distribution of the -- schools -- node with normalization", prob_schoolsNewNormal)
        
        Update_value = np.random.choice(['bad','good'],p=[prob_schoolsNewNormal['bad'],prob_schoolsNewNormal['good']])
        return Update_value

    def probability_age(self, nonevidList, inpevidenceList):

        '''Calculate the probability distribution for age node based on Markov Blanket and then
           normalizing it to get it within the 0-1 range '''
        
#        print ("\nCalculating the probability of the randomly selected --", "age", "--\n")
        _ = self.markov_Blanket('age')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
#        print(totalList['age'])
        #print ("Complete List -- ", totalList)

        prob_ageNewnoNorm, prob_ageNewNormal = {}, {}
        for age_nodeOption in self.ageOptions:
            prob_ageNewnoNorm[age_nodeOption] = self.CPT_age(age_nodeOption, totalList['location'])*self.CPT_location(totalList['location'],totalList['amenities'],totalList['neighborhood'])\
                                                *self.CPT_price(totalList['price'], totalList['location'], age_nodeOption,totalList['schools'],totalList['size'])*\
                                                self.CPT_size(totalList['size'])*self.CPT_schools(totalList['schools'], totalList['children'])
        
            
        #p(location|age,age)*p()
        summ = sum(list(prob_ageNewnoNorm.values()))
        for key in list(prob_ageNewnoNorm.keys()):
            prob_ageNewNormal[key] = prob_ageNewnoNorm[key]/summ

        #print ("Probability distribution of the -- age -- node without normalization", prob_ageNewnoNorm)
        #print ("Probability distribution of the -- age -- node with normalization", prob_ageNewNormal)
        
        Update_value = np.random.choice(['old','new'],p=[prob_ageNewNormal['old'],prob_ageNewNormal['new']])
        return Update_value

    def probability_price(self, nonevidList, inpevidenceList):

        '''Calculate the probability distribution for price node based on Markov Blanket and then
           normalizing it to get it within the 0-1 range '''
        
        #print ("\nCalculating the probability of the randomly selected --", "price", "--\n")
        _ = self.markov_Blanket('price')

        nonevidList.update(inpevidenceList) #Concatenates nonevidence and input evidence lists
        totalList = nonevidList
#        print(totalList['price'])
        #print ("Complete List -- ", totalList)

        prob_priceNewnoNorm, prob_priceNewNormal = {}, {}
        for price_nodeOption in self.priceOptions:
            prob_priceNewnoNorm[price_nodeOption] = self.CPT_price(price_nodeOption, totalList['location'], totalList['age'],totalList['schools'],totalList['size'])*\
                                                    self.CPT_age(totalList['age'], totalList['location'])*self.CPT_location(totalList['location'],totalList['amenities'],totalList['neighborhood'])\
                                                    *self.CPT_size(totalList['size'])*self.CPT_schools(totalList['schools'], totalList['children'])
            
        #p(location|price,price)*p()
        summ = sum(list(prob_priceNewnoNorm.values()))
        for key in list(prob_priceNewnoNorm.keys()):
            prob_priceNewNormal[key] = prob_priceNewnoNorm[key]/summ

        #print ("Probability distribution of the -- price -- node without normalization", prob_priceNewnoNorm)
        #print ("Probability distribution of the -- price -- node with normalization", prob_priceNewNormal)
        
        Update_value = np.random.choice(['cheap','ok','expensive'],p=[prob_priceNewNormal['cheap'],prob_priceNewNormal['ok'],prob_priceNewNormal['expensive']])
        return Update_value
    
    def calculate_probability(self):
        
        checkingNode = self.QueryNode
        IgnoredSamples = int(self.numSampleIgnr/(len(self.allNodes) - len(list(self.inpevidenceList))))
        if checkingNode == 'amenities':
            
            ''' Magic line: ignores the given no. of initial observations (defaults to 0 if value not given) 
                     [note: list does not need to be sorted, but can be sorted if required]'''
            stateList = Counter({k: self.amenitiesStates[k] for k in list(self.amenitiesStates)[IgnoredSamples:]}.values())
            
            #Normalizing the readings to obtain probability of state
            lots = stateList['lots']/float(stateList['lots']+stateList['little'])
            little = stateList['little']/float(stateList['lots']+stateList['little'])
            print('Probabilities of states of node -Amenities- are --> \nlots: ',lots,'  \nlittle: ',little)
            
        elif checkingNode == 'neighborhood':
            
            stateList = Counter({k: self.neighborhoodStates[k] for k in list(self.neighborhoodStates)[IgnoredSamples:]}.values())
            bad = stateList['bad']/float(stateList['bad']+stateList['good'])
            good = stateList['good']/float(stateList['bad']+stateList['good'])
            print('Probabilities of states of node -neighborhood- are --> \nbad: ',bad,'  \ngood: ',good)
        
        elif checkingNode == 'location':
            
            stateList = Counter({k: self.locationStates[k] for k in list(self.locationStates)[IgnoredSamples:]}.values())
            bad = stateList['bad']/float(stateList['bad']+stateList['good']+stateList['ugly'])
            good = stateList['good']/float(stateList['bad']+stateList['good']+stateList['ugly'])
            ugly = stateList['ugly']/float(stateList['bad']+stateList['good']+stateList['ugly'])
            print('Probabilities of states of node -location- are --> \nbad: ',bad,'  \ngood: ',good, '\nugly: ',ugly)

        elif checkingNode == 'children':
            
            stateList = Counter({k: self.childrenStates[k] for k in list(self.childrenStates)[IgnoredSamples:]}.values())
            bad = stateList['bad']/float(stateList['bad']+stateList['good'])
            good = stateList['good']/float(stateList['bad']+stateList['good'])
            print('Probabilities of states of node -children- are --> \nbad: ',bad,'  \ngood: ',good)
            
        elif checkingNode == 'size':
            
            stateList = Counter({k: self.sizeStates[k] for k in list(self.sizeStates)[IgnoredSamples:]}.values())
            small = stateList['small']/float(stateList['small']+stateList['medium']+stateList['large'])
            medium = stateList['medium']/float(stateList['small']+stateList['medium']+stateList['large'])
            large = stateList['large']/float(stateList['small']+stateList['medium']+stateList['large'])
            print('Probabilities of states of node -size- are --> \nsmall: ',small,'  \nmedium: ',medium, '\nlarge: ',large)
            
        elif checkingNode == 'schools':
            stateList = Counter({k: self.schoolsStates[k] for k in list(self.schoolsStates)[IgnoredSamples:]}.values())
            bad = stateList['bad']/float(stateList['bad']+stateList['good'])
            good = stateList['good']/float(stateList['bad']+stateList['good'])
            print('Probabilities of states of node -schools- are --> \nbad: ',bad,'  \ngood: ',good)
 
        elif checkingNode == 'age':
            stateList = Counter({k: self.ageStates[k] for k in list(self.ageStates)[IgnoredSamples:]}.values())
            old = stateList['old']/float(stateList['old']+stateList['new'])
            new = stateList['new']/float(stateList['old']+stateList['new'])
            print('Probabilities of states of node -children- are --> \nold: ',old,'  \nnew: ',new) 
            
        elif checkingNode == 'price':
            stateList = Counter({k: self.priceStates[k] for k in list(self.priceStates)[IgnoredSamples:]}.values())
            cheap = stateList['cheap']/float(stateList['cheap']+stateList['ok']+stateList['expensive'])
            ok = stateList['ok']/float(stateList['cheap']+stateList['ok']+stateList['expensive'])
            expensive = stateList['expensive']/float(stateList['cheap']+stateList['ok']+stateList['expensive'])
            print('Probabilities of states of node -price- are --> \ncheap: ',cheap,'  \nok: ',ok, '\nexpensive: ',expensive)

        
#Defining the main function that creates the object for the Class and does some shit - This needs to be structured better

def main():
    start = time.time()
    gibbs_obj = Gibbs()
    nonevidList, inpevidenceList, numUpdates, numSampleIgnr, QueryNode, = gibbs_obj.nodeValueSetting()

    allValues_noevidList = list(nonevidList.values())
    allValues_noevidList = list(nonevidList.keys())
    allValues_length = len(allValues_noevidList)

    allValues_evidList = list(inpevidenceList.values())
    allValues_evidList = list(inpevidenceList.keys())

    print ("Non evidence List", nonevidList)
    print ("Evidence List", inpevidenceList)

    print ("Query Node is -- ", QueryNode)
    print ("Number of updates  -- ", numUpdates)
    print ("Number of initial samples to ignore -- ", numSampleIgnr )
    print ("---------------\n")
    
    '''Iterating over the non-evidence nodes for length of times defined by user'''
    IterationTimes = numUpdates

    print("Iterating over non-evidence nodes for ",IterationTimes," iterations, Updating probabilities of non-evidence nodes and sampling the states\n")
    print ("---------------\n")
    
    '''[NOTE: Since we iterate through all nodes, number of updates will be divided by No. of evidence nodes]'''
    
    #No. of updates can be calculated as follows:
    UpdateNum = int(IterationTimes/allValues_length)
    
    start = time.time()
    for counter in range(0,UpdateNum):

        #Temporary nodelist to keep track of iterated nodes and prevent multiple iterations in a single loop
        iterated_nodeList = {} 
        
        #Checks length of temporary list to match it with the main list, equal length means iteration over all non-evidence nodes is complete
        while (len(list(iterated_nodeList)) != allValues_length):
            #counter += 1
            #Select a random node based on random probability, eventually iterate through all nodes with the loop
            randomNode = allValues_noevidList[random.randint(0, len(allValues_noevidList)-1)]
            if not randomNode in iterated_nodeList.keys():
                iterated_nodeList[randomNode] = 'status: iterated'
                
                if randomNode== 'amenities':
                    
                    #Generates new "state" for the selected node, based on updated probabilities
                    New_Node_Val = gibbs_obj.probability_amenities(nonevidList, inpevidenceList)
                    
                    #Update the main dictionary containing all node states
                    nonevidList[randomNode] = New_Node_Val
                    
                    #Append the result to a cumulitive dictionary to keep track of all states received so far
                    gibbs_obj.amenitiesStates[counter] = New_Node_Val
                    
                elif randomNode== 'neighborhood':
                    New_Node_Val = gibbs_obj.probability_neighborhood(nonevidList, inpevidenceList)
                    nonevidList[randomNode] = New_Node_Val
                    gibbs_obj.neighborhoodStates[counter] = New_Node_Val
                    
                elif randomNode== 'location':
                    New_Node_Val = gibbs_obj.probability_location(nonevidList, inpevidenceList)
                    nonevidList[randomNode] = New_Node_Val
                    gibbs_obj.locationStates[counter] = New_Node_Val
                    
                elif randomNode== 'size':
                    New_Node_Val = gibbs_obj.probability_size(nonevidList, inpevidenceList)
                    nonevidList[randomNode] = New_Node_Val
                    gibbs_obj.sizeStates[counter] = New_Node_Val         
                    
                elif randomNode== 'children':
                    New_Node_Val = gibbs_obj.probability_children(nonevidList, inpevidenceList)
                    nonevidList[randomNode] = New_Node_Val
                    gibbs_obj.childrenStates[counter] = New_Node_Val
                    
                elif randomNode== 'schools':
                    New_Node_Val = gibbs_obj.probability_schools(nonevidList, inpevidenceList)
                    nonevidList[randomNode] = New_Node_Val
                    gibbs_obj.schoolsStates[counter] = New_Node_Val
                    
                elif randomNode== 'age':
                    New_Node_Val = gibbs_obj.probability_age(nonevidList, inpevidenceList)
                    nonevidList[randomNode] = New_Node_Val
                    gibbs_obj.ageStates[counter] = New_Node_Val

                elif randomNode== 'price':
                    New_Node_Val = gibbs_obj.probability_price(nonevidList, inpevidenceList)
                    nonevidList[randomNode] = New_Node_Val
                    gibbs_obj.priceStates[counter] = New_Node_Val
        #print('Iteration: ',counter,'\n')
        
    #Final function which calculates probability based on states recorded of the query node
    gibbs_obj.calculate_probability()
    end = time.time()
    print('\nElapsed time - ',end-start,' seconds')

main()

    





