/*
 *
 */
package machinelearningcorsework;

import static java.lang.Math.sqrt;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;

/**
 * a implementation of a linear perception with added features
 * @author Ed
 */
public class EnhancedLinearPerceptron extends LinearPerceptron {
    
   private boolean standardised=false;//are values to be standardise
   private boolean offLine=false;//do you do offline update
   private double[] mean ; //mean data
   private double[] standrdDeviation ;//standrdDeviation of data
   private boolean maybeUseOffline =false; //see if need to test to see if use offLine
   
    public void setMaybeUseOffline(boolean toSet){
        maybeUseOffline=toSet;
    }
    
    public boolean getMaybeUseOffline(){
        return maybeUseOffline;
    }
   
    public void setStandardised(boolean toSet){
        standardised=toSet;
    }
    
    public boolean getStandardised(){
        return standardised;
    }
    
    public void setOffLine(boolean toSet){
        offLine=toSet;
    }
    
    public boolean getOffLine(){
        return offLine;
    }
    /**
     *  to question if should use offline update
     * Guesses by splitting test data in half and train two models one 
     * using online other using offline then testing against other half of data
     * Uses which ever is most successful
     * @param data being tested
     * @return 
     */
   private boolean useOffLine(Instances data){
       
       Instances test=new Instances(data,0); //data being split in to testing and training data 
       Instances train = new Instances(data,0);  
      
       double[] offlineWeight= new double[data.numAttributes()-1]; //set weights for online/offline
       double[] onlineWeight= new double[data.numAttributes()-1];
  
       for(int x =0;x<(data.numInstances()/2);x++){ //load half data in to each 
           train.add(data.instance(x));
           test.add(data.instance(x+(data.numInstances()/2)));
       }
      
       
       
       //change instances in to double arrays
       double [][] doubledDataTrain = new double[train.numInstances()][train.numAttributes()]; 
       double [][] doubledDataTest = new double[test.numInstances()][test.numAttributes()];
       
       for(int instanceNum =0;instanceNum<train.numInstances();instanceNum++){
            double[] temp=train.instance(instanceNum).toDoubleArray();
            for( int abNumber =0; abNumber<temp.length;abNumber++){
                doubledDataTrain[instanceNum][abNumber]=temp[abNumber];
            }
        }
       
       for(int instanceNum =0;instanceNum<test.numInstances();instanceNum++){
            double[] temp=test.instance(instanceNum).toDoubleArray();
            for( int abNumber =0; abNumber<temp.length;abNumber++){
                doubledDataTest[instanceNum][abNumber]=temp[abNumber];
            }
        }
       
       //load in starting weights
       double[] weights = new double[data.numAttributes()];
       for(int x =0;x<weights.length;x++){
           weights[x]=1;
       }
       offlineWeight=offlineLearning(doubledDataTrain,weights);//get offline weights
       onlineWeight=onlineLearning(doubledDataTrain,train,1,weights);//get online weights
       //see if offlien or online was more successful at defineing a weight
       int offlineSuccess=0;
       int onlineSuccess=0;
       for(int x =0; x<doubledDataTest.length;x++){
           double[] currentInstance =doubledDataTest[x];
           double toReturn=0;
           for (int temp =0;temp<currentInstance.length-1;temp++){
               toReturn+=(offlineWeight[temp])*(currentInstance[temp]);              
           }
            toReturn=roundNumberAnswers(toReturn);
            if (toReturn==currentInstance[currentInstance.length-1]){
                offlineSuccess++;
            }
       }
       
       
       for(int x =0; x<doubledDataTest.length;x++){
           double[] currentInstance =doubledDataTest[x];
           double toReturn=0;
           for (int temp =0;temp<currentInstance.length-1;temp++){
               toReturn+=(onlineWeight[temp])*(currentInstance[temp]);              
           }
            toReturn=roundNumberAnswers(toReturn);
            if (toReturn==currentInstance[currentInstance.length-1]){
                onlineSuccess++;
            }
       }
       
       //return which value return most correct guesses
       if (offlineSuccess>onlineSuccess){
           // System.out.println("Decided on using offline");
            return true;
       }else{
           // System.out.println("Decided on using online");
           return false;
       }
       
   }
    /**
     *  preforms offline Learning on doubledData
     * @param doubledData data to learn form 
     * @param weightsLearn starting weight
     * @return new weight
     */
    private double[] offlineLearning( double[][] doubledData,double[] weightsLearn){
           boolean success;
           boolean fin ; 
           int roationNumber =0;
          // System.out.println(Arrays.toString(weightsLearn));
           double[] changeInWeight=new double[weightsLearn.length]; 
           while(true){
               for(int x = 0;x<changeInWeight.length;x++){
                   changeInWeight[x]=0;
               }
              // System.out.println("Rotation though data Number "+roationNumber);
             
               roationNumber++;
               for (int x=0;x<doubledData.length;x++){
                   
                   double answer=guessAnswer(doubledData[x],weightsLearn);
                   for(int y=0;y<weightsLearn.length;y++){
                       changeInWeight[y]+=0.5*1*(doubledData[x][doubledData[x].length-1]-answer)*doubledData[x][y]; //set change in weight
                   }
                   
               }
               fin=true;
               for(int y=0;y<weightsLearn.length;y++){
                   if(changeInWeight[y]!=0){ //see if change needed
                       if (bias==true){
                            weightsLearn[y]+=changeInWeight[y]+1; //apply change in weight
                       }else{
                            weightsLearn[y]+=changeInWeight[y];
                       }
                       fin=false;//if changed need to rotate again
                   }
               }
               if(fin==true){ //if ending 
                //   System.out.println("No change in weights");
                //   System.out.println("Success");
                //   System.out.println("Current Weights = "+Arrays.toString(weights));
                   success=true;
                   break;
               }
               if ((roationNumber*doubledData.length)>maxNumberTurns){
                //   System.out.println(roationNumber*doubledData.length);
                //    System.out.println("\nFAIL TO REACH SOLUTION\nREACHED MAX NUMBER OF TERMS");
                    success=true;
                    break;
               }
               
        }
           
           return weightsLearn;
    }
    /**
     * 
     * @param doubledData data to learn form 
     * @param data data instances form 
     * @param learningFactor learning factor
     * @param weightsLearn starting weight
     * @return new weight
     */
    private double[] onlineLearning( double[][] doubledData,Instances data,int learningFactor,double[] weightsLearn){
        boolean success;
        double[][] previousValues=preLoadBuildClassifierValues(data);
        int toEndInt= 0;//number times value hasnt changed
        int stepNumber=1;
        int count=0;//Which value is being worked on 
        while(true){
          //      System.out.println("Step Number "+stepNumber);
          //      System.out.println("Current Weights = "+Arrays.toString(weightsLearn));
                stepNumber++;
                if (count==data.numInstances()){ //to loop on instances
                    count=0;
                }
                doStep(doubledData,count,learningFactor);//do one step of 
                weightsLearn=weights;
                boolean checkIfSame=true;
                for(int temp =0;temp<weightsLearn.length;temp++){
                    //weights[temp] =roundNumber(weights[temp],dp);
                        if (weightsLearn[temp]!=previousValues[temp][count]){ // see if weights are same as last time
                              checkIfSame=false;
                              previousValues[temp][count]=weightsLearn[temp];
                              toEndInt++;//add one to length of chain of same records  
                           }
                 }
                 if(checkIfSame==true){ //if same
               
                     toEndInt++; //add one to length
                     if(toEndInt>data.numInstances()){//success break out
                          success=true;
                   //       System.out.println("\nSUCCESS");
                         // System.out.println("Final Weight = "+Arrays.toString(weightsLearn));
                        break;
                    }else{//fail break out stuck in loop
                    success=false;
                 //   System.out.println("\nFAIL\nNO WEIGHT POSSIBLE FOR DATA SET LOOPING");
                    break;
                        }
                }
            if (maxNumberTurns<stepNumber){
            //   System.out.println("\nFAIL TO REACH SOLUTION\nREACHED MAX NUMBER OF TERMS");
               success=true;
               break;
            }
           
                count++;
            }
            return weightsLearn;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception  {
       int learningFactor=1;
       boolean success=false;
       boolean useOfflineQuestion = false;
       
       if (maybeUseOffline==true){
            useOfflineQuestion =useOffLine(data);
       }
      
      
       double[][] doubledData= new double[data.numInstances()][data.numAttributes()];
      
       if (standardised==true){
           doubledData=standardisedUnits(data);//standardise data
       }else{
           //losd none standardise in to doubledData
           for(int instanceNum =0;instanceNum<data.numInstances();instanceNum++){
            double[] temp=data.instance(instanceNum).toDoubleArray();
            for( int abNumber =0; abNumber<temp.length;abNumber++){
                doubledData[instanceNum][abNumber]=temp[abNumber];
            }
        }
       }
        
       if(offLine==true || (maybeUseOffline && useOfflineQuestion)){ //If offline
           preLoadBuildClassifierValues(data);
           weights=offlineLearning(doubledData,weights);
       }else{
          
          weights=onlineLearning(doubledData,data,learningFactor,weights);
       }
       
    }
    
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
       double[] instanceDoubleArray=instance.toDoubleArray();
       double answer=0;
       if(standardised==true){ //If standised weight standised instance
            for (int temp =0;temp>instanceDoubleArray.length-1;temp++){
                instanceDoubleArray[temp]=(instanceDoubleArray[temp]-mean[temp])/standrdDeviation[temp];
            }
       }
      
        answer=guessAnswer(instanceDoubleArray,weights);
       
       return roundNumberAnswers(answer);

    }
    
    
    /**
     * Calucate mean of a array of doubles
     * @param data
     * @return double
     */
    private double calucateMean(double[] data){
        double toReturn=0;
        for(int x =0; x<data.length;x++){
            toReturn=(data[x]+toReturn);
        }
        toReturn=toReturn/data.length;
        return toReturn;
    }
    /**
     * Calucate Standard Deviation of a array of doubles
     * @param data
     * @return double
     */
    private double calucateStandardDeviation (double[] data){
       double sd = 0;
       double mean2=calucateMean(data);
        for (int i = 0; i < data.length; i++){
            //System.out.println(((data[i] - mean2)*(data[i] - mean2)));
        sd += ((data[i] - mean2)*(data[i] - mean2));
        }
        
        sd=sd/data.length;
        
        return sqrt(sd);
    }
    
    public static double roundNumber(double value, int places) {
    double scale = Math.pow(10, places);
    return Math.round(value * scale) / scale;
}
    
    /**
     * standardises the data given
     * @param data
     * @return 
     */
    public double[][] standardisedUnits(Instances data){
        double[][] doubledData= new double[data.numAttributes()][data.numInstances()];
        //Read data in so its attribute to instance
        for(int instanceNum =0;instanceNum<data.numInstances();instanceNum++){
            double[] temp=data.instance(instanceNum).toDoubleArray();
            for( int abNumber =0; abNumber<temp.length;abNumber++){
                doubledData[abNumber][instanceNum]=temp[abNumber];
            }
        }
        
        
        mean=new double[data.numAttributes()];
        standrdDeviation=new double[data.numAttributes()];
        
        double[][] standisedData= new double[data.numInstances()][data.numAttributes()];
        //standerise all data
        for( int abNumber =0; abNumber<data.numAttributes()-1;abNumber++){
            mean[abNumber] =calucateMean(doubledData[abNumber]);
            standrdDeviation[abNumber] = calucateStandardDeviation(doubledData[abNumber]);
           //  System.out.println("Calulated mean");
           // System.out.println(Arrays.toString(mean));
         //   System.out.println("Calulated standrdDeviation");
           // System.out.println(Arrays.toString(standrdDeviation));
            for(int instanceNum =0;instanceNum<data.numInstances();instanceNum++){
                standisedData[instanceNum][abNumber]=(doubledData[abNumber][instanceNum]-mean[abNumber])/standrdDeviation[abNumber];
                //standisedData[instanceNum][abNumber]=roundNumber(standisedData[instanceNum][abNumber],dp);
            }
        }
        double[] answers=doubledData[doubledData.length-1];
        for(int x = 0;x<answers.length;x++){
            standisedData[x][data.numAttributes()-1]=answers[x];
        }
     //     System.out.println("Calulated standised values");  
  //        System.out.println(Arrays.deepToString(standisedData));
        return standisedData;
}
    
    
}
