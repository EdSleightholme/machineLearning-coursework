
package machinelearningcorsework;

import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Ed
 */
public class LinearPerceptronEnsemble implements Classifier{
    private int size =50;//number 
    private int popOfAtubutes =50;//amount of attubutes to remove as perecentage
    ArrayList<EnhancedLinearPerceptron> listClassifers =new ArrayList<>();
    ArrayList<Remove> listFilters =new ArrayList<>();
    
    public int getPopOfAtubutes(){
         return popOfAtubutes;
    }
    
    public void setPopOfAtubutes(int toSet){
        popOfAtubutes=toSet;
    }
    
    public int getSize(){
         return size;
    }
    
    public void setSize(int toSet){
        size=toSet;
    }
    
    ArrayList<EnhancedLinearPerceptron> getListClassifers(){
        return listClassifers;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        //load in EnhancedLinearPerceptron with random bits data and random conditions
        for(int x =0;x<size;x++){
 
            Random rand = new Random();
            EnhancedLinearPerceptron toAdd =new EnhancedLinearPerceptron(); //to be added to listClassifers
            //set classifers settings
            int  choose = rand.nextInt(100) + 1;
            
            if (choose>50){
                toAdd.setBias(true);
            }
            
            choose = rand.nextInt(100) + 1;
            if (choose>50){
                toAdd.setOffLine(true);
            }
            
            choose = rand.nextInt(100) + 1;
            if (choose>50){
                toAdd.setStandardised(false);
            }
            
            choose = rand.nextInt(100) + 1;
            if (choose>50){
            toAdd.setMaybeUseOffline(true);
            }
           
            //sets data to be used to train toAdd classifer
            Instances toTrain= new Instances(data); 
            
            //filter to remove certain attubutes
            Remove getRideAbb =new Remove();

            int[] attributesToRemove = new int[(int)(data.numAttributes()*(double)popOfAtubutes/100)+1];
            //randomly set attubutes to remove
            for(int y =0;y<attributesToRemove.length;y++){
                int c=rand.nextInt(100) + 1;
                choose = rand.nextInt(data.numAttributes()-1);
                attributesToRemove[y]=choose;
            }
            
            getRideAbb.setAttributeIndicesArray(attributesToRemove);
            getRideAbb.setInputFormat(data);
            
            toTrain=Remove.useFilter(data,getRideAbb);
            //train classifers
            toAdd.buildClassifier(toTrain);
            //add classsifer to list of classifers
            listClassifers.add(toAdd);
            listFilters.add(getRideAbb);
            }
           
           } 
        
        
        
        
    

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] toReturn = new double [instance.classAttribute().numValues()];
        Remove filterToUse;
        for(int x =0;x<size;x++){
            filterToUse=listFilters.get(x);
            filterToUse.input(instance);
            Instance toUse =filterToUse.output();
            
            EnhancedLinearPerceptron lookingAt = listClassifers.get(x);
            
            if(listClassifers.get(x).classifyInstance(toUse)==-1){
                toReturn[0]++;
            }else{
                toReturn[1]++;
            }

        }
        double maxValue =0;
        for(int x =0;x<toReturn.length;x++){
            if(toReturn[x]>toReturn[(int)maxValue]){
                maxValue=x;
            }
        }
        return maxValue;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
         double[] toReturn = new double [instance.classAttribute().numValues()];
        for(int x =0;x<size;x++){
            EnhancedLinearPerceptron lookingAt = listClassifers.get(x);
            
            toReturn[(int) listClassifers.get(x).classifyInstance(instance)]++;

        }
        return toReturn;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
