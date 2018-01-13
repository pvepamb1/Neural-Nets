
public class Connection {

	private Neuron input;
	private Neuron output;
	private float weight;
	private int id;
	private int layer;
	private float newWeight;

	public Connection(Neuron input, Neuron output, float weight, float newWeight, int id, int layer) {
		this.input = input;
		this.output = output;
		this.weight = weight;
		this.newWeight = newWeight;
		this.id = id;
		this.layer = layer;
	}

	public Neuron getInput() {
		return input;
	}

	public void setInput(Neuron input) {
		this.input = input;
	}

	public Neuron getOutput() {
		return output;
	}

	public void setOutput(Neuron output) {
		this.output = output;
	}

	public float getWeight() {
		return weight;
	}

	public void setWeight(float weight) {
		this.weight = weight;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public int getLayer() {
		return layer;
	}

	public void setLayer(int layer) {
		this.layer = layer;
	}

	public float getNewWeight() {
		return newWeight;
	}

	public void setNewWeight(float newWeight) {
		this.newWeight = newWeight;
	}

}
