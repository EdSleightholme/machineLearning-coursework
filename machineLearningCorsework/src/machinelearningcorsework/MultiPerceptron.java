
package machinelearningcorsework;

import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Ed
 */
public class MultiPerceptron implements Classifier {
    private ArrayList<EnhancedLinearPerceptron> switchs;

    
    public static double log2(double num){
        return (Math.log(num)/Math.log(2));
    }
    

    public double[][] changeAnswer(double[][] toChange,int temp){
        for(int x=0;x<toChange.length;x++){
            if(toChange[x][toChange[x].length-1]%temp==0){
                toChange[x][toChange[x].length-1]=0;
            }else{
                toChange[x][toChange[x].length-1]=1;
            }
        }
        return toChange;
    }
    
    
    public void buildClassifier(Instances i) throws Exception {
        
        //convert i in to a double array 
        double[][] iDouble=new double[i.numInstances()][i.numAttributes()];
        
        for(int instan =0;instan<i.numInstances();instan++){
            for(int abb =0;abb<i.numAttributes();abb++){
                iDouble[instan][abb]=0;
            }
        }
        
        
        //get number possible class values 
        int numberAnswers =i.classAttribute().numValues()+1;

        //make number of EnhancedLinearPerceptron for num possible values
        for(int x =1;x<(log2(numberAnswers) + 2 );x++){
            double[][] temp =iDouble;
            temp=changeAnswer(temp,2*x);
            EnhancedLinearPerceptron toAdd = new EnhancedLinearPerceptron();

        }

        
            //build perceptron on data set up to only 1 and 0s in right points 
            
        //END
    
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

   

    

    
    
    
    
}
