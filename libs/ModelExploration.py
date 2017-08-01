"""
Model exploration

    A few correct labels at random
    A few incorrect labels at random
    The most correct labels of each class (ie those with highest probability that are correct)
    The most incorrect labels of each class (ie those with highest probability that are incorrect)
    The most uncertain labels (ie those with probability closest to 0.5).
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.random import random, permutation
from sklearn.metrics import confusion_matrix
import utils
from utils import plots, get_batches, plot_confusion_matrix, get_data
from keras.preprocessing import image


np.set_printoptions(precision=10, linewidth=100)

def AddRow(gs, row, img_paths, titles, header):
    ims = [plt.imread(path) for path in img_paths]
    for i in range(len(ims)):
         ax = plt.subplot(gs[row, i]) # TODO matplotlib image
         ax.set_title(titles[i])
         ax.set_xticks([])
         ax.set_yticks([])
         ax.imshow(ims[i])  
         if (i == 0):
            ax.set_ylabel(header, rotation=90, size='large')


def Explore(class_indicies_dict, actual_labels, predicted_labels, predicted_probs, validation_file_paths):

    gs = gridspec.GridSpec(2 + len(class_indicies_dict), 4) #, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)

    # Reverse
    mapping = {v:k for k,v in class_indicies_dict.items()}

    # Number of images to view for each visualization task
    n_view = 4

    ##########################################################
    # A few correct labels at random
    ##########################################################

    header = "Random correct"
    all_correct = np.where(predicted_labels == actual_labels)[0] # returns a tuple with one value!?  Note that this is the index!  not the
                                                                 # value.
    idx = permutation(all_correct)[:n_view]
    titles = [mapping[predicted_labels[i]] + "=" + str(predicted_probs[i,predicted_labels[i]]) for i in idx]
    AddRow(gs, 0, [validation_file_paths[i] for i in idx], titles, header)
    
    ##########################################################
    # A few incorrect labels at random
    ##########################################################

    header = "Random incorrect"
    all_incorrect = np.where(predicted_labels != actual_labels)[0]
    idx = permutation(all_incorrect)[:n_view]
    titles = [mapping[predicted_labels[i]] + "=" + str(predicted_probs[i,predicted_labels[i]]) for i in idx]
    AddRow(gs, 1, [validation_file_paths[i] for i in idx], titles, header)
    
    ##########################################################
    # The images of label X we are most confident and correct
    ##########################################################

    for lbl in range(0,len(mapping)):
        header = "Most correct " + mapping[lbl]
        # shorter list with indicies into original data
        all_correct = np.where((predicted_labels == lbl) & (predicted_labels == actual_labels))[0] # indicies

        predicted_probs_subset = predicted_probs[all_correct,lbl] # same size as all_correct

                                                                                   # Note that np.argsort gives the indicies for only the shorter correct
                                                                                   # cats!  we cant use these indicies in the original data.
        most_correct = np.argsort(predicted_probs_subset)[::-1][:n_view] # returns the indicies that would sort the array, [::-1] = copy of list in
                                                                                   # reverse order
        idx = all_correct[most_correct] # Back to indicies of dataset
        titles = [mapping[predicted_labels[i]] + "=" + str(predicted_probs[i,predicted_labels[i]]) for i in idx]
        AddRow(gs, 2 + lbl, [validation_file_paths[i] for i in idx], titles, header)
    
    # Same by minimizing dog probabilities - BS: sum of probabilities > 1
    # Looks different, does not seem to be good!  Note that the probabilities
    # do not sum up exactly to 1
    
    
    # Intermitted plot
    plt.show()
    gs = gridspec.GridSpec(2 + len(class_indicies_dict), 4) #, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
   
    ##########################################################
    # The images of label X we are most confident but wrong
    ##########################################################

    for lbl in range(0,len(mapping)):
    
        
        header = "Most confident " + mapping[lbl] + " but incorrect)"
        incorrect = np.where((predicted_labels == lbl) & (predicted_labels != actual_labels))[0]
        most_incorrect = np.argsort(predicted_probs[incorrect,lbl])[::-1][:n_view]
        idx = incorrect[most_incorrect]
        titles = [mapping[predicted_labels[i]] + "=" + str(predicted_probs[i,predicted_labels[i]]) for i in idx]
        AddRow(gs, 0 + lbl, [validation_file_paths[i] for i in idx], titles, header)
   
    ##########################################################
    # The most uncertain labels (ie those with probability closest to 0.5).
    ##########################################################

    header = "Most uncertain (around 0.5)"
    most_uncertain = np.argsort(np.abs(predicted_probs[:,1] - 0.5))[:n_view] # if one is 0.5, so is the other
    idx = most_uncertain
    titles = [mapping[predicted_labels[i]] + "=" + str(predicted_probs[i,predicted_labels[i]]) for i in idx]
    AddRow(gs, 2, [validation_file_paths[i] for i in idx], titles, header)
    
    # Intermitted plot
    plt.show()

    # confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels)
    plot_confusion_matrix(cm, class_indicies_dict)
    plt.show()
    
