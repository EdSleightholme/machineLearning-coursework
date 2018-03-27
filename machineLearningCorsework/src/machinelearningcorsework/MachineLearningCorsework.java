/*
TODO 
 fix standised units
 fix classifyInstance fort ensemble for more than 2 possible answers

Test buildClassifier 
    Enhanced basic online 
        Runs correct                    tick
        Ends When hits max num terms    tick
        Runs with bias                  tick
    Enhanced standised online           tick
        Runs correct                    tick
        Ends When hits max num terms    tick
        Runs with bias                  tick
    Enhanced basic offline
        Runs correct                    tick
        Ends When hits max num terms    tick
        Runs with bias                  tick
    Enhanced standised offline          tick
        Runs correct                    
        Ends When hits max num terms    
        Runs with bias     
    Enhanced basic choose offline       tick
        Runs correct                    tick
        Ends When hits max num terms    tick
        Runs with bias                  tick
    Ensemble basic 
        Runs correct                    
        Ends When hits max num terms    
        Runs with bias                  

Test classifyInstance 
    Enhanced basic online
        Runs correct                    tick
        Runs with bias                  tick
    Enhanced standised online           X
        Runs correct                     
        Runs with bias     
    Enhanced basic offline              
        Runs correct                    tick
        Runs with bias                  tick
    Enhanced standised offline          X
        Runs correct                    
        Runs with bias                  
    Enhanced basic choose offline       
        Runs correct                    tick
        Runs with bias                  tick
    Ensemble basic 
        Runs correct                    
        Runs with bias                  

Test distributionForInstance            
    Ensemble basic                      tick
        Runs correct                    tick
        Runs with bias                  tick

to CSV
    TPR
    FPR
    FNR
    TNR
    Accuracy
    Balanced Accuracy
    Sensitivity
    Specificity
    Recall
    Precision
    F1
*/   
package machinelearningcorsework;




import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Ed 
 */
public class MachineLearningCorsework {

    
    public static void main(String[] args) throws FileNotFoundException {
        
        //runOneArffFile("trains.arff");
        
        ArrayList<String> listFiles = new ArrayList<String>();
        ArrayList<String> toPrint = new  ArrayList<String>();
        double[] toWrite;
        File file = new File("listArffFiles.csv");
        Scanner input = new Scanner(file);
            // hashNext() loops line-by-line
            while(input.hasNext()){
                //read single line, put in string
                String data = input.next();
                listFiles.add(data);

            }
           
            // after loop, close scanner
            input.close();
        
       //get hit miss data
       for(int x=0;x<listFiles.size()-1;x++){
            System.out.println("TESTING\t"+listFiles.get(x));
            toPrint.add((runOneArffFile(listFiles.get(x))));
            System.out.println("DONE\t"+listFiles.get(x));
       }
       
       PrintWriter pw = new PrintWriter(new File("output.csv"));
       StringBuilder sb = new StringBuilder();

       //header names
       sb.append("name of file,");
       sb.append("num instance,");
       sb.append("num abbs,");
       sb.append("normal hit");
       sb.append(",,,,,,,,");
       sb.append("standised hit");
       sb.append(",,,,,,,,");
       sb.append("bias hit");
       sb.append(",,,,,,,,");
       sb.append("offline hit");
       sb.append(",,,,,,,,");
       sb.append("maybe hit");
      sb.append(",,,,,,,,");
       sb.append("stan+bias hit");
       sb.append(",,,,,,,,");
       sb.append("offline+bias hit");
      sb.append(",,,,,,,,");
       sb.append("maybe+bias hit");
       sb.append(",,,,,,,,");
       sb.append("offline+stan hit");
      sb.append(",,,,,,,,");
       sb.append("maybe+stan hit");
      sb.append(",,,,,,,,");
       sb.append("stan+bias+offline hit");
       sb.append(",,,,,,,,");
       sb.append("stan+bias+maybe hit");
       sb.append(",,,,,,,,");
       sb.append("enemble10 hit");
        sb.append(",,,,,,,,");
       sb.append("enemble50 hit");
        sb.append(",,,,,,,,");
       sb.append("enemble100 hit,\n");
       sb.append(",,,");
       for(int x=0;x<15;x++){
           sb.append(addsubHeader());

       }
       sb.append('\n');
       //System.out.println(toPrint.get(0)[1]);
       for(int x=0;x<toPrint.size();x++){
           sb.append(listFiles.get(x));
           sb.append(toPrint.get(x)); 
           sb.append('\n');
       }
       pw.write(sb.toString());
       pw.close();
       // System.out.println(Arrays.toString(toPrint.get(0)));
    }
   /* TPR
    FPR
    FNR
    TNR
    Accuracy
    Balanced Accuracy
    Sensitivity
    Specificity
    Recall
    Precision
    F1*/
    public static String addsubHeader(){
    StringBuilder sb = new StringBuilder();  
      sb.append("TPR");   
      sb.append(',');   
      sb.append("FPR");   
      sb.append(',');   
      sb.append("FNR");   
      sb.append(',');   
      sb.append("TNR");   
      sb.append(',');   
      sb.append("Accuracy");   
      sb.append(',');   
      sb.append("Balanced Accuracy");   
      sb.append(',');   
      sb.append("Precision");   
      sb.append(',');   
      sb.append("F1");   
      sb.append(','); 
      return sb.toString();
    }
     
    
    
     
        
     
    public static String runEmembleSetSize(String title,int sizeOfEnsemble){
            
        double answer=0;
        int[] hit =new int[15];
        int[] miss =new int[15];
        double trueP=0;
        double trueN=0;
        double falseP=0;
        double falseN=0;
        double[] tpr =new double[15];     
        double[] fpr =new double[15];     
        double[] fnr =new double[15];     
        double[] tnr =new double[15];     
        double[] accuracy =new double[15];     
        double[] balAccuracy =new double[15];     
        LinearPerceptronEnsemble classiferEnemble = new LinearPerceptronEnsemble();
        classiferEnemble.setSize(sizeOfEnsemble);
        double[] precision =new double[15];     
        double[] f1 =new double[15];     
        Instances allData;
        allData=arffFileLoader(title);
        allData.setClassIndex(allData.numAttributes()-1);
        Random rand = new Random(); 
        allData.randomize(rand);
        int numInstances=allData.numInstances();
       Instances trainData =new Instances(allData,0) ;
       Instances testData=new Instances(allData,0) ;
       
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classiferEnemble.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
    
    }
    public static String runOneArffFile(String title){
        double answer=0;
        int[] hit =new int[15];
        int[] miss =new int[15];
        double trueP=0;
        double trueN=0;
        double falseP=0;
        double falseN=0;
        double[] tpr =new double[15];     
        double[] fpr =new double[15];     
        double[] fnr =new double[15];     
        double[] tnr =new double[15];     
        double[] accuracy =new double[15];     
        double[] balAccuracy =new double[15];     
        
        double[] precision =new double[15];     
        double[] f1 =new double[15];     
        Instances allData;
        allData=arffFileLoader(title);
        allData.setClassIndex(allData.numAttributes()-1);
        Random rand = new Random(); 
        allData.randomize(rand);
        int numInstances=allData.numInstances();
       Instances trainData =new Instances(allData,0) ;
       Instances testData=new Instances(allData,0) ;
        for(int x=0;x<numInstances-1;x++){
            
            if ((numInstances/2)<x){
                trainData.add(allData.instance(x));
            }else{
                testData.add(allData.instance(x));
            }
            
        }
        //System.out.println(allData.numInstances());
        
        
        
       EnhancedLinearPerceptron classifer = new EnhancedLinearPerceptron();
       classifer.setMaxNumberTurns(trainData.numInstances()*15);
       try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
       
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
               
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[0]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[0]=trueP/(trueP+falseP);
        tnr[0]=trueN/(trueN+falseN);
        fpr[0]=falseP/(trueP+falseP);
        fnr[0]=falseN/(trueN+falseN);
        balAccuracy[0]=(tpr[0]+tnr[0])/2;
        precision[0]=trueP/(trueP+falseP);
        f1[0]=2*((tpr[0]*precision[0])/(tpr[0]+precision[0]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
        System.out.println("test stanised data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setStandardised(true);
       try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
       
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[1]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[1]=trueP/(trueP+falseP);
        tnr[1]=trueN/(trueN+falseN);
        fpr[1]=falseP/(trueP+falseP);
        fnr[1]=falseN/(trueN+falseN);
        balAccuracy[1]=(tpr[1]+tnr[1])/2;
        precision[1]=trueP/(trueP+falseP);
        f1[1]=2*((tpr[1]*precision[1])/(tpr[1]+precision[1]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
       
        
        
        
        
        System.out.println("test bias data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setBias(true);
         
       try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
      
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                  
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[2]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[2]=trueP/(trueP+falseP);
        tnr[2]=trueN/(trueN+falseN);
        fpr[2]=falseP/(trueP+falseP);
        fnr[2]=falseN/(trueN+falseN);
        balAccuracy[2]=(tpr[2]+tnr[2])/2;
        precision[2]=trueP/(trueP+falseP);
        f1[2]=2*((tpr[2]*precision[2])/(tpr[2]+precision[2]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
         System.out.println("test offline data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setOffLine(true);
         
       try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
      
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                 
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[3]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[3]=trueP/(trueP+falseP);
        tnr[3]=trueN/(trueN+falseN);
        fpr[3]=falseP/(trueP+falseP);
        fnr[3]=falseN/(trueN+falseN);
        balAccuracy[3]=(tpr[3]+tnr[3])/2;
        precision[3]=trueP/(trueP+falseP);
        f1[3]=2*((tpr[3]*precision[3])/(tpr[3]+precision[3]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
          System.out.println("test maybe offline data");
            classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setMaybeUseOffline(true);
         try {
           //  System.out.println(trainData.toString());
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
       
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                  
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[4]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[4]=trueP/(trueP+falseP);
        tnr[4]=trueN/(trueN+falseN);
        fpr[4]=falseP/(trueP+falseP);
        fnr[4]=falseN/(trueN+falseN);
        balAccuracy[4]=(tpr[4]+tnr[4])/2;
        precision[4]=trueP/(trueP+falseP);
        f1[4]=2*((tpr[4]*precision[4])/(tpr[4]+precision[4]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
          System.out.println("test stan+bia data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setBias(true);
         classifer.setStandardised(true);
         try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
      
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                  
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[5]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[5]=trueP/(trueP+falseP);
        tnr[5]=trueN/(trueN+falseN);
        fpr[5]=falseP/(trueP+falseP);
        fnr[5]=falseN/(trueN+falseN);
        balAccuracy[5]=(tpr[5]+tnr[5])/2;
        precision[5]=trueP/(trueP+falseP);
        f1[0]=2*((tpr[5]*precision[5])/(tpr[5]+precision[5]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
         System.out.println("test off+bia data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setBias(true);
         classifer.setOffLine(true);
         try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
     
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                 
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[6]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[6]=trueP/(trueP+falseP);
        tnr[6]=trueN/(trueN+falseN);
        fpr[6]=falseP/(trueP+falseP);
        fnr[6]=falseN/(trueN+falseN);
        balAccuracy[6]=(tpr[6]+tnr[6])/2;
        precision[6]=trueP/(trueP+falseP);
        f1[6]=2*((tpr[6]*precision[6])/(tpr[6]+precision[6]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
         System.out.println("test may+bia data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setBias(true);
         classifer.setMaybeUseOffline(true);
         try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
     
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                 
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[7]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[7]=trueP/(trueP+falseP);
        tnr[7]=trueN/(trueN+falseN);
        fpr[7]=falseP/(trueP+falseP);
        fnr[7]=falseN/(trueN+falseN);
        balAccuracy[7]=(tpr[7]+tnr[7])/2;
        precision[7]=trueP/(trueP+falseP);
        f1[7]=2*((tpr[7]*precision[7])/(tpr[7]+precision[7]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
         System.out.println("test off+stan data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setOffLine(true);
         classifer.setStandardised(true);
         try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
      
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[8]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[8]=trueP/(trueP+falseP);
        tnr[8]=trueN/(trueN+falseN);
        fpr[8]=falseP/(trueP+falseP);
        fnr[8]=falseN/(trueN+falseN);
        balAccuracy[8]=(tpr[8]+tnr[8])/2;
        precision[8]=trueP/(trueP+falseP);
        f1[8]=2*((tpr[8]*precision[8])/(tpr[8]+precision[8]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
          System.out.println("test maybe+stan data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setMaybeUseOffline(true);
         classifer.setStandardised(true);
         try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
      
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
               if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[9]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[9]=trueP/(trueP+falseP);
        tnr[9]=trueN/(trueN+falseN);
        fpr[9]=falseP/(trueP+falseP);
        fnr[9]=falseN/(trueN+falseN);
        balAccuracy[9]=(tpr[9]+tnr[9])/2;
        precision[9]=trueP/(trueP+falseP);
        f1[9]=2*((tpr[9]*precision[9])/(tpr[9]+precision[9]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
        
         System.out.println("test offline+stan+bias data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setOffLine(true);
         classifer.setStandardised(true);
         classifer.setBias(true);
         try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
      
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
     
        accuracy[10]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[10]=trueP/(trueP+falseP);
        tnr[10]=trueN/(trueN+falseN);
        fpr[10]=falseP/(trueP+falseP);
        fnr[10]=falseN/(trueN+falseN);
        balAccuracy[10]=(tpr[10]+tnr[10])/2;
        precision[10]=trueP/(trueP+falseP);
        f1[10]=2*((tpr[10]*precision[10])/(tpr[10]+precision[10]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
       
        
          System.out.println("test maybe+stan+bias data");
         classifer = new EnhancedLinearPerceptron();
         classifer.setMaxNumberTurns(trainData.numInstances()*15);
         classifer.setOffLine(true);
         classifer.setStandardised(true);
         classifer.setBias(true);
         try {
            classifer.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
      
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classifer.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
              if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[11]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[11]=trueP/(trueP+falseP);
        tnr[11]=trueN/(trueN+falseN);
        fpr[11]=falseP/(trueP+falseP);
        fnr[11]=falseN/(trueN+falseN);
        balAccuracy[11]=(tpr[11]+tnr[11])/2;
        precision[11]=trueP/(trueP+falseP);
        f1[11]=2*((tpr[11]*precision[11])/(tpr[11]+precision[11]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
        
         System.out.println("test linear emble size 10 ");
        LinearPerceptronEnsemble classiferEnemble = new LinearPerceptronEnsemble();
        classiferEnemble.setSize(10);
         try {
            classiferEnemble.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
     
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classiferEnemble.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
               if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[12]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[12]=trueP/(trueP+falseP);
        tnr[12]=trueN/(trueN+falseN);
        fpr[12]=falseP/(trueP+falseP);
        fnr[12]=falseN/(trueN+falseN);
        balAccuracy[12]=(tpr[12]+tnr[12])/2;
        precision[12]=trueP/(trueP+falseP);
        f1[12]=2*((tpr[12]*precision[12])/(tpr[12]+precision[12]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
        
        
        System.out.println("test linear emble size 50 ");
        classiferEnemble = new LinearPerceptronEnsemble();
        classiferEnemble.setSize(50);
         try {
            classiferEnemble.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
     
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classiferEnemble.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[13]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[13]=trueP/(trueP+falseP);
        tnr[13]=trueN/(trueN+falseN);
        fpr[13]=falseP/(trueP+falseP);
        fnr[13]=falseN/(trueN+falseN);
        balAccuracy[13]=(tpr[13]+tnr[13])/2;
        precision[13]=trueP/(trueP+falseP);
        f1[13]=2*((tpr[13]*precision[13])/(tpr[13]+precision[13]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        
        
           System.out.println("test linear emble size 100 ");
        classiferEnemble = new LinearPerceptronEnsemble();
        classiferEnemble.setSize(100);
         try {
            classiferEnemble.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
        }
     
        
        for(int x=0;x<testData.numInstances();x++){
            
            try {
               answer =classiferEnemble.classifyInstance(testData.instance(x));
            } catch (Exception ex) {
                Logger.getLogger(MachineLearningCorsework.class.getName()).log(Level.SEVERE, null, ex);
            }
            double[] doubleArray=testData.instance(x).toDoubleArray();
            if(answer==doubleArray[testData.instance(x).classIndex()]){
                if(answer==1){
                   trueP++; 
                }else{
                   trueN++;
                }
            }else{
               if(answer==1){
                   falseP++; 
                }else{
                   falseN++;
                }
            }
        
        }
        
        
        accuracy[14]=(trueP+trueN)/(trueP+trueN+falseP+falseN);
        tpr[14]=(double)trueP/(double)(trueP+falseP);
        tnr[14]=(double)trueN/(double)(trueN+falseN);
        fpr[14]=(double)falseP/(double)(trueP+falseP);
        fnr[14]=(double)falseN/(double)(trueN+falseN);
        balAccuracy[14]=(double)((double)tpr[14]+(double)tnr[14])/2;
        precision[14]=(double)trueP/(double)(trueP+falseP);
        f1[14]=2*(((double)tpr[14]*precision[14])/(double)(tpr[14]+precision[14]));
        trueP=0;
        trueN=0;
        falseP=0;
        falseN=0;
        
        System.out.println("FINAL RESULTS");
       // System.out.println(Arrays.toString(hit));
       // System.out.println(Arrays.toString(miss));
        StringBuilder toReturn=new StringBuilder();
        toReturn.append(',');
        toReturn.append(allData.numInstances());
        toReturn.append(',');
        toReturn.append(allData.numAttributes());
       toReturn.append(',');

       for(int x=0;x<15;x++){
           toReturn.append(tpr[x]);
           toReturn.append(',');
           toReturn.append(tnr[x]);
           toReturn.append(',');
           toReturn.append(fpr[x]);
           toReturn.append(',');
           toReturn.append(fnr[x]);
           toReturn.append(',');
           toReturn.append(accuracy[x]);
           toReturn.append(',');
           toReturn.append(balAccuracy[x]);
           toReturn.append(',');
           toReturn.append(precision[x]);
           toReturn.append(',');
           toReturn.append(f1[x]);
           toReturn.append(',');
       }
       
        
        //System.out.println(Arrays.toString(toReturn));
       
        
        return toReturn.toString();
    }
    
    public static Instances arffFileLoader(String filePath) {
        Instances train;
        try{
            FileReader reader = new FileReader(filePath);
            train = new Instances(reader);
            return(train);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return null;
    }
}
