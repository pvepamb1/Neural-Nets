package com.neural;

public interface NetworkTester
{
    void setTestStrategy(TestStrategy testStrategy);
    void setModel(Model model);
    void setDataLoader(DataLoader dataLoader);
    void test();
}
