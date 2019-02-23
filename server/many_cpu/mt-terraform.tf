variable "image" {
  default = "mt-baby-worker-img"
}

// Configure the Google Cloud provider
provider "google" {
  project     = "stop-harassing-staging"
  region      = "us-east1-c"
}

resource "google_compute_instance" "mathtastic-worker" {
  count                     = "1"
  name                      = "mt-cpu-worker"
  machine_type              = "n1-standard-24"
  zone                      = "us-east1-c"
  tags                      = ["cpu-compute"]

  // We use this key instead of the `startup-script` key to force instance recreation when the contents change
  metadata_startup_script = "${file("${path.module}/mt-terraform-startup.sh")}"
  metadata {
    shutdown-script = "${file("${path.module}/mt-terraform-shutdown.sh")}"
  }
  // Use an optimized Tensorflow build that requires Broadwell or newer, from: https://github.com/mind/wheels/
  min_cpu_platform = "Intel Broadwell"
  // Stopping for update is required to set the cpu platform: 
  // https://www.terraform.io/docs/providers/google/r/compute_instance.html#min_cpu_platform
  allow_stopping_for_update = true

  // Disk storage
  boot_disk {
    initialize_params {
      // Custom built image with CUDA and such already installed
      image = "${var.image}"
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral IP
    }
  }

  service_account {
    scopes = ["userinfo-email", "compute-ro", "storage-full" ]
  }

  # ------------------------------------------------
  # GPU Configuration options
  # ------------------------------------------------
  scheduling {
    # Instances with guest accelerators do not support live migration.
    on_host_maintenance = "TERMINATE"
    preemptible = true
    automatic_restart = false
  }
}
