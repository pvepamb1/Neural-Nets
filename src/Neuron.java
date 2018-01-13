import java.util.ArrayList;

public class Neuron {

	private ArrayList<Connection> conn;
	private OutputWrap output;
	private Float expectedOut;
	private int layer;
	private float bias;

	public Neuron(OutputWrap output, Float expectedOut, int layer, float bias) {
		this.output = output;
		this.expectedOut = expectedOut;
		this.layer = layer;
		this.bias = bias;
	}

	public ArrayList<Connection> getConn() {
		return conn;
	}

	public void setConn(ArrayList<Connection> conn) {
		this.conn = conn;
	}

	public ArrayList<Float> getWeights() {
		ArrayList<Float> temp = new ArrayList<Float>();
		for (Connection c : conn)
			temp.add(c.getWeight());
		return temp;
	}

	public ArrayList<OutputWrap> getInputs() {
		ArrayList<OutputWrap> temp = new ArrayList<OutputWrap>();
		for (Connection c : conn)
			temp.add(c.getInput().getOutput());
		return temp;
	}

	public OutputWrap getOutput() {
		return output;
	}

	public void setOutput(OutputWrap output) {
		this.output = output;
	}

	public int getLayer() {
		return layer;
	}

	public void setLayer(int layer) {
		this.layer = layer;
	}

	public Float getExpectedOut() {
		return expectedOut;
	}

	public void setExpectedOut(Float expectedOut) {
		this.expectedOut = expectedOut;
	}

	public float getBias() {
		return bias;
	}

	public void setBias(float bias) {
		this.bias = bias;
	}

}
