package com.neural;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

public class MnistImageGenerator {

    public static void main(String[] args) throws IOException {
        generateImage("/Users/prasenna/Downloads/train-images.idx3-ubyte", "/Users/prasenna/Downloads/train-labels.idx1-ubyte", "/Users/prasenna/Downloads/mnist-images/train");
    }

    public static void generateImage(String dataFilePath, String labelFilePath, String outputPath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        assert numberOfItems == numberOfLabels;

        System.out.println("Generating images..");

        BufferedImage image = new BufferedImage(nCols, nRows, BufferedImage.TYPE_BYTE_GRAY);

        for(int i = 0; i < numberOfItems; i++) {
            int label = labelInputStream.readUnsignedByte();
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    int pixel = dataInputStream.readUnsignedByte();
                    image.setRGB(c, r, pixel | (pixel << 8) | (pixel << 16));
                }
            }

            String outputFileName = label + "_" + i + ".png";
            ImageIO.write(image, "png", new File(outputPath+ "/" + outputFileName));
        }
        dataInputStream.close();
        labelInputStream.close();
    }
}
