
package machinelearningcorsework;

import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * a impementation of a linear Perceptron for a single simple 1,0 answer
 * @author Ed
 */
public class LinearPerceptron implements Classifier  {
    protected double[] weights ;
    protected boolean bias =false;
    protected int maxNumberTurns =10000;
    
    public void setMaxNumberTurns(int toSet){
        maxNumberTurns=toSet;
    }
    
    public int getMaxNumberTurns(){
        return maxNumberTurns;
    }
    
    public void setBias(boolean toSet){
        bias=toSet;
    }
    
    public boolean getBias(){
        return bias;
    }
    /**
     * 
     * @param instanceLookingAt which instance in a instances you are working on
     * @return guess of what the answer should be using current weights only does 0 or 1
     */
    protected double guessAnswer(double[] instanceLookingAt,double[] weight){
        double toReturn=0;
        for (int temp =0;temp<instanceLookingAt.length-1;temp++){
               toReturn+=(weight[temp])*(instanceLookingAt[temp]);              
           }
        toReturn=roundNumberAnswers(toReturn);
        return toReturn;
    }
    /**
     *  adjusts Weights with new values
     * @param instanceLookingAt which instance in a instances you are working on
     * @param learningFactor 
     * @param answerForInstance weight of guessed anwser
     * @param weightGuess =-1,+1,0
     */
    protected void adjustWeight(double[] instanceLookingAt,int learningFactor,double answerForInstance, double weightGuess){
        for(int temp =0;temp<weights.length;temp++){//adjust weights
            if (bias==true){
                    weights[temp]= (double) (weights[temp]+0.5*learningFactor*(answerForInstance-weightGuess)*instanceLookingAt[temp])+1;
            }else{
                weights[temp]= (double) (weights[temp]+0.5*learningFactor*(answerForInstance-weightGuess)*instanceLookingAt[temp]);
            }
        }
    }
    /**
     * do calucation for 1 instance 
     * @param data array of doubles containing a double array of instance values
     * @param count number of instnace for calucation in data
     * @param learningFactor 
    */
    protected void doStep(double[][] data,int count,int learningFactor){
           double[] instanceLookingAt=data[count];//get instance we are looking at 
           double answerForInstance =instanceLookingAt[instanceLookingAt.length-1];//that result of instance 
           
           double answerGuessed =0;//weight guessed by algorithm 
           answerGuessed=guessAnswer(instanceLookingAt,weights);
           
           if(answerGuessed!=answerForInstance){ //if guess wrong
             //  System.out.println("AJUST WEIGHTS");
               adjustWeight(instanceLookingAt,learningFactor,answerForInstance,answerGuessed);
           }
        //   System.out.println("\n");
    }
    
    /**
     * rounds weightGuess to +1,-1 or 0
     * @param weightGuess
     * @return -1,+1 or 0
     */
    protected double roundNumberAnswers(double weightGuess){
         if (weightGuess<0){
           return -1;
       }else if(weightGuess>0){
           return +1;
       }else{
           return 0;
       }
    }
    
    /**
     * clears all values in class ready for new classfier
     * @param data instances being used to train model
     * @return a new double[data.numAttributes()][data.numInstances()]; all values set =0
     */
     protected double[][] preLoadBuildClassifierValues(Instances data){
       weights=new double[data.numAttributes()-1];//weight of desion making
       for(int temp =0;temp<weights.length;temp++){ //preloads 1 in to all desion making values
           weights[temp]=1;
       }
       double[][] previousValues= new double[data.numAttributes()][data.numInstances()];//previous values at each instance
         
       for(int temp2 =0;temp2<previousValues.length;temp2++){
            for(int temp =0;temp<previousValues[0].length;temp++){//preload 0 in to each value
                previousValues[temp2][temp]=0;
            }
       }
       
       return previousValues;
     }
    /**
     * builds classifer for data
     * @param data instances to be Classified
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances data) throws Exception  {
        
       int learningFactor=1;
       boolean success=false;
       double[][] previousValues=preLoadBuildClassifierValues(data); //sort out all preloading values
       
       
       double[][] doubledData= new double[data.numInstances()][data.numAttributes()]; //to hold all instance data
       
       for(int instanceNum =0;instanceNum<data.numInstances();instanceNum++){ //load all instance data in to doubledData
            double[] temp=data.instance(instanceNum).toDoubleArray();
            for( int abNumber =0; abNumber<temp.length;abNumber++){
                doubledData[instanceNum][abNumber]=temp[abNumber];
            }
        }
       
       int toEndInt= 0;//number times value hasnt changed
       int stepNumber=1;//how many steps have been done by algorithm 
       int count=0;//Which value is being worked on 
       while(true){
         //  System.out.println("Step Number "+stepNumber);
           //System.out.println("Current Weights = "+Arrays.toString(weights));
           
           stepNumber++;
           if (count==data.numInstances()){ //to loop on instances
               count=0;
           }
           
           doStep(doubledData,count,learningFactor); //do one step on instance in 
           
           boolean checkIfSame=true;//see if time to end algorithm
           for(int temp =0;temp<weights.length;temp++){
                   if (weights[temp]!=previousValues[temp][count]){ // see if weights have changed
                       checkIfSame=false;
                       previousValues[temp][count]=weights[temp];
                       toEndInt++;//add one to length of chain of same records  
                   }
           }
           if(checkIfSame==true){ //if same
               
               toEndInt++; //add one to length
               if(toEndInt>data.numInstances()){//if no chnge found break out
                   success=true;
                  // System.out.println("\nSUCCESS");
               //    System.out.println("Final Weight = "+Arrays.toString(weights));
                   break;
               }else{//fail break out stuck in loop
               success=false;
              // System.out.println("\nFAIL\nNO WEIGHT POSSIBLE FOR DATA SET");
               break;
                }
           }
           if (maxNumberTurns<stepNumber){
              // System.out.println("\nFAIL TO REACH SOLUTION\nREACHED MAX NUMBER OF TERMS");
               success=true;
               break;
           }
           
           count++;
       }
       if (success==false){ //finsihed but no solution found
           throw new Exception("message");
       }
       
    }

    
   
    
   /**
    * 
    * @param instance to have anwser guessed
    * @return +1/-1 or 0
    * @throws Exception 
    */
    @Override
    public double classifyInstance(Instance instance) throws Exception {
       double[] instanceDoubleArray=instance.toDoubleArray();
       double answer=0;
       answer=guessAnswer(instanceDoubleArray,weights);//guess answer of instance
       return roundNumberAnswers(answer);//

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
