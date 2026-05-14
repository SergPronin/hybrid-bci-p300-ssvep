using System;
using System.Globalization;
using System.IO;
using System.IO.Ports;
using System.Threading;
using System.Windows.Forms;
using System.Xml;

namespace Migalka
{
    public partial class MyForm : Form
    {
        private int[] Counts;
        private static readonly string InitFileName = "Settings.xml";

        private ComboBox cbMode;

        public MyForm()
        {
            InitializeComponent();

            // ===== РЕЖИМ =====
            cbMode = new ComboBox();
            cbMode.DropDownStyle = ComboBoxStyle.DropDownList;
            cbMode.Items.AddRange(new object[] { "Постоянный", "Пакетный" });
            cbMode.SelectedIndex = 0;

            cbMode.Width = 120;
            cbMode.Left = 10;
            cbMode.Top = 2;

            cbMode.SelectedIndexChanged += cbMode_SelectedIndexChanged;

            panel1.Controls.Add(cbMode);

            // ===== ТАБЛИЦА =====
            dgFrequency.Rows.Add(6);

            ((DataGridViewComboBoxColumn)dgFrequency.Columns[1]).Items.Add("0");
            for (int i = 1; i <= 500; i++)
                ((DataGridViewComboBoxColumn)dgFrequency.Columns[1]).Items.Add(
                    (1000.0f / i).ToString().Replace(',', '.')
                );

            for (int i = 0; i < dgFrequency.Rows.Count; i++)
            {
                dgFrequency.Rows[i].Cells[0].Value = i + 1;
                dgFrequency.Rows[i].Cells[1].Value = "0";
            }

            Counts = new int[8];

            btnCOMPort_Click(btnCOMPort, new EventArgs());
            LoadInit();
        }

        // ================= РЕЖИМ (С ПОЛНЫМ REFRESH) =================
        private void cbMode_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!serialPort1.IsOpen)
                return;

            try
            {
                // 1. отправляем режим
                string modeCmd = cbMode.SelectedIndex == 0 ? "M 0" : "M 1";
                serialPort1.WriteLine(modeCmd);

                Thread.Sleep(50);

                // 2. ОБЯЗАТЕЛЬНО пересылаем ВСЕ частоты заново (фикс ламп)
                string s = string.Format("{0} {1} {2} {3} {4} {5}",
                    dgFrequency.Rows[0].Cells[1].Value,
                    dgFrequency.Rows[1].Cells[1].Value,
                    dgFrequency.Rows[2].Cells[1].Value,
                    dgFrequency.Rows[3].Cells[1].Value,
                    dgFrequency.Rows[4].Cells[1].Value,
                    dgFrequency.Rows[5].Cells[1].Value
                );

                serialPort1.WriteLine(s);

                textBox1.AppendText($"\r\n=> {modeCmd}\r\n=> {s}");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        // ================= LOAD =================
        private void LoadInit()
        {
            if (!File.Exists(InitFileName))
                return;

            XmlDocument xmlDoc = new XmlDocument();
            xmlDoc.Load(InitFileName);

            XmlNode xmlNode = xmlDoc.GetElementsByTagName("Settings")[0];
            if (xmlNode == null) return;

            xmlNode = xmlNode["Blink"];
            if (xmlNode == null) return;

            for (int i = 0; i < dgFrequency.Rows.Count; i++)
            {
                XmlElement el = xmlNode["Blink" + i];
                if (el == null) return;

                dgFrequency.Rows[i].Cells[1].Value = el.GetAttribute("Frequency");
            }
        }

        // ================= SAVE =================
        private void SaveInit()
        {
            XmlDocument xmlDoc = new XmlDocument();

            if (File.Exists(InitFileName))
                xmlDoc.Load(InitFileName);

            XmlNode root = xmlDoc["Settings"];

            if (root == null)
            {
                root = xmlDoc.CreateElement("Settings");
                xmlDoc.AppendChild(root);
            }

            XmlElement blink = root["Blink"];

            if (blink == null)
            {
                blink = xmlDoc.CreateElement("Blink");
                root.AppendChild(blink);
            }

            for (int i = 0; i < dgFrequency.Rows.Count; i++)
            {
                XmlElement el = blink["Blink" + i];

                if (el == null)
                {
                    el = xmlDoc.CreateElement("Blink" + i);
                    blink.AppendChild(el);
                }

                el.SetAttribute("Frequency", dgFrequency.Rows[i].Cells[1].Value.ToString());
            }

            xmlDoc.Save(InitFileName);
        }

        // ================= КНОПКА =================
        private void button1_Click(object sender, EventArgs e)
        {
            if (!serialPort1.IsOpen)
            {
                if (cbCOMPort.SelectedIndex == -1)
                {
                    MessageBox.Show("Задайте порт");
                    return;
                }

                try
                {
                    serialPort1.PortName = cbCOMPort.SelectedItem.ToString();
                    serialPort1.Open();

                    Thread.Sleep(500);

                    string s = string.Format("{0} {1} {2} {3} {4} {5}",
                        dgFrequency.Rows[0].Cells[1].Value,
                        dgFrequency.Rows[1].Cells[1].Value,
                        dgFrequency.Rows[2].Cells[1].Value,
                        dgFrequency.Rows[3].Cells[1].Value,
                        dgFrequency.Rows[4].Cells[1].Value,
                        dgFrequency.Rows[5].Cells[1].Value
                    );

                    serialPort1.WriteLine(s);
                    Thread.Sleep(100);

                    textBox1.Text = "=> " + s + "\r\n<=" + serialPort1.ReadExisting();

                    cbCOMPort.Enabled = false;
                    button1.Text = "Остановить";
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
            else
            {
                try
                {
                    serialPort1.WriteLine("0 0 0 0 0 0");
                    Thread.Sleep(200);

                    serialPort1.Close();
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }

                cbCOMPort.Enabled = true;
                button1.Text = "Задать";
            }
        }

        // ================= COM =================
        private void btnCOMPort_Click(object sender, EventArgs e)
        {
            string old = cbCOMPort.Text;

            string[] ports = SerialPort.GetPortNames();

            cbCOMPort.Items.Clear();

            foreach (string p in ports)
                cbCOMPort.Items.Add(p);

            if (cbCOMPort.Items.Count == 1)
                cbCOMPort.SelectedIndex = 0;
            else
                cbCOMPort.SelectedIndex = cbCOMPort.Items.IndexOf(old);
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            try
            {
                if (serialPort1.IsOpen)
                {
                    serialPort1.WriteLine("0 0 0 0 0 0");
                    Thread.Sleep(200);
                    serialPort1.Close();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            SaveInit();
        }

        private void cbCOMPort_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (serialPort1.IsOpen)
                serialPort1.Close();
        }
    }
}