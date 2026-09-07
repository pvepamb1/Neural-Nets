package com.neural.concurrent;

public class ErrorStat implements Cloneable
{
    private double errorTotal;
    private double previousError = Integer.MAX_VALUE;
    private double minError = Integer.MAX_VALUE;
    private double errorRiseFromPreviousCount;
    private double errorRiseFromMinCount;

    public double getErrorTotal()
    {
        return errorTotal;
    }

    public void setErrorTotal(double errorTotal)
    {
        this.errorTotal = errorTotal;
    }

    public double getPreviousError()
    {
        return previousError;
    }

    public void setPreviousError(double previousError)
    {
        this.previousError = previousError;
    }

    public double getMinError()
    {
        return minError;
    }

    public void setMinError(double minError)
    {
        this.minError = minError;
    }

    public double getErrorRiseFromPreviousCount()
    {
        return errorRiseFromPreviousCount;
    }

    public void setErrorRiseFromPreviousCount(double errorRiseFromPreviousCount)
    {
        this.errorRiseFromPreviousCount = errorRiseFromPreviousCount;
    }

    public double getErrorRiseFromMinCount()
    {
        return errorRiseFromMinCount;
    }

    public void setErrorRiseFromMinCount(double errorRiseFromMinCount)
    {
        this.errorRiseFromMinCount = errorRiseFromMinCount;
    }

    @Override
    public ErrorStat clone()
    {
        try
        {
            return (ErrorStat) super.clone();
        }
        catch (CloneNotSupportedException e)
        {
            throw new RuntimeException(e);
        }
    }

}
