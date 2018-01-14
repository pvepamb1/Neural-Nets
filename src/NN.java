import java.util.ArrayList;
import java.util.Scanner;

public class NN {

	private static int hiddenNeurons = 2;
	private static int outputNeurons = 2;
	private static ArrayList<Float> inputArray = new ArrayList<Float>();
	private static ArrayList<Neuron> neuronList = new ArrayList<Neuron>();
	private static ArrayList<Connection> weightArray = new ArrayList<Connection>();
	private static int layer = 1;
	private static float eta = 0.5f;
	private static float[] biasArray = { 0.35f, 0.60f };

	public static void main(String[] args) {
		inputArray.add(0.05f);
		inputArray.add(0.10f);
		createNeurons();
		createConnections();
		setNeurons();
		System.out.println("Iterations?");
		Scanner s = new Scanner(System.in);
		int a = s.nextInt();
		s.close();
		for (int i = 0; i <= a; i++) {
			for (Neuron n : neuronList) {
				if (n.getLayer() != 0)
					calculate(n);
			}
			backpropogate();
			update();
		}
		error();
		test();
	}

	public static void createNeurons() {
		for (float f : inputArray) {
			neuronList.add(new Neuron(new OutputWrap(f), null, 0, 0.0f));
		}
		for (int i = 0; i < hiddenNeurons; i++) {
			neuronList.add(new Neuron(new OutputWrap(0), null, layer, biasArray[0]));
		}
		layer++;
		for (int i = 0; i < outputNeurons; i++) {
			neuronList.add(new Neuron(new OutputWrap(0), null, layer, biasArray[1]));
		}
	}

	private static void createConnections() {
		int id = 0;
		for (int i = 0; i < layer; i++) {
			for (Neuron n : neuronList) {
				if (n.getLayer() == i) {
					for (Neuron n2 : neuronList) {
						if (n2.getLayer() == i + 1) {
							weightArray.add(new Connection(n, n2, (float) Math.random(), 0.0f, id, i));
							id++;
						}
					}
				}
			}
		}
	}

	public static void setNeurons() {
		int i = 0;
		for (Neuron n : neuronList) {
			if (n.getLayer() != 0) {
				ArrayList<Float> weights = new ArrayList<Float>();
				ArrayList<OutputWrap> inputs = new ArrayList<OutputWrap>();
				ArrayList<Connection> connections = new ArrayList<Connection>();
				for (Connection c : weightArray) {
					if (c.getOutput() == n) {
						connections.add(c);
						weights.add(c.getWeight());
						inputs.add(c.getInput().getOutput());
					}
				}
				n.setConn(connections);
			}
			if (n.getLayer() == layer) {
				switch (i) {
				case 0:
					n.setExpectedOut(0.01f);
					i++;
					break;
				case 1:
					n.setExpectedOut(0.99f);
					i++;
					break;
				}
			}
		}
	}

	public static void calculate(Neuron n) {
		float total = 0;
		for (int i = 0; i < n.getWeights().size(); i++) {
			total += n.getWeights().get(i) * n.getInputs().get(i).getOutput();
		}
		total += n.getBias();
		total = (float) (1 / (1 + Math.exp(-total)));
		n.getOutput().setOutput(total);
	}

	public static void error() {
		float total = 0.0f;
		for (Neuron n : neuronList) {
			if (n.getLayer() == layer) {
				total += (float) 1 / 2 * (Math.pow(n.getExpectedOut() - n.getOutput().getOutput(), 2));
			}
		}
		System.out.println(total);
	}

	public static void backpropogate() {

		int temp = layer - 1;
		while (temp >= 0) {
			for (Connection c : weightArray) {
				if (c.getLayer() == temp) {
					if (temp == layer - 1) {
						float err = c.getOutput().getOutput().getOutput() - c.getOutput().getExpectedOut();
						float err2 = c.getOutput().getOutput().getOutput()
								* (1 - (c.getOutput().getOutput().getOutput()));
						float totalErr = err * err2 * c.getInput().getOutput().getOutput();
						c.setNewWeight(c.getWeight() - eta * totalErr);
					} else {
						float totalerr1 = 0.0f;
						Neuron n = c.getOutput();
						for (Connection c2 : weightArray) {
							if (c2.getLayer() == temp + 1) {
								if (c2.getInput() == n) {
									float err = c2.getOutput().getOutput().getOutput()
											- c2.getOutput().getExpectedOut();
									float err2 = c2.getOutput().getOutput().getOutput()
											* (1 - (c2.getOutput().getOutput().getOutput()));
									float totalErr = err * err2 * c2.getWeight();
									totalerr1 += totalErr;
								}
							}
						}
						float t2 = c.getOutput().getOutput().getOutput()
								* (1 - (c.getOutput().getOutput().getOutput()));
						float t3 = c.getInput().getOutput().getOutput();
						float t = totalerr1 * t2 * t3;
						c.setNewWeight(c.getWeight() - eta * t);
					}
				}
			}
			temp--;
		}
	}

	public static void update() {
		for (Connection c : weightArray) {
			c.setWeight(c.getNewWeight());
		}
	}

	public static void test() {

		for (Neuron n : neuronList)
			if (n.getLayer() == layer)
				System.out.println(n.getOutput().getOutput());

		neuronList.get(0).getOutput().setOutput(7.15f);
		neuronList.get(1).getOutput().setOutput(8.10f);

		for (Neuron n : neuronList)
			if (n.getLayer() != 0)
				calculate(n);

		for (Neuron n : neuronList)
			if (n.getLayer() == layer)
				System.out.println(n.getOutput().getOutput());
	}
}
