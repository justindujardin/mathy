variable "image" {
  default = "ml-dojo-img"
}

// Configure the Google Cloud provider
provider "google" {
  project     = "stop-harassing-staging"
  region      = "us-east1-c"
}

resource "google_compute_instance" "ml-dojo-worker" {
  count = "1"
  name         = "ml-dojo-worker"
  machine_type = "custom-1-6144"
  zone         = "us-east1-c"

  metadata_startup_script = "#!/usr/bin/env bash\n/usr/local/bin/ml-dojo-terraform-init"

  tags = ["gpu-compute"]

  boot_disk {
    initialize_params {
      size = "20"
      image = "${var.image}"
      
    }
  }

  network_interface {
    network = "default"

    access_config {
      // Ephemeral IP
    }
  }
  metadata {
  }

  scheduling {
    # Instances with guest accelerators do not support live migration.
    on_host_maintenance = "TERMINATE"
  }
  # Configure with a GPU
  guest_accelerator {
    count = 1
    type = "nvidia-tesla-k80"
  }

  service_account {
    scopes = ["userinfo-email", "compute-ro", "storage-ro"]
  }
}
