variable "image" {
  default = "ml-dojo-img"
}

// Configure the Google Cloud provider
provider "google" {
  project     = "stop-harassing-staging"
  region      = "us-east1-c"
}

resource "google_compute_instance" "ml-dojo-worker" {
  count                     = "1"
  name                      = "ml-dojo-worker"
  machine_type              = "custom-1-6144"
  zone                      = "us-east1-c"
  metadata_startup_script   = "#!/usr/bin/env bash\n/usr/local/bin/ml-dojo-terraform-init"
  tags                      = ["gpu-compute"]
  // Use an optimized Tensorflow build that from: https://github.com/mind/wheels/
  min_cpu_platform = "Intel Broadwell"
  // Stopping for update is required to set the cpu platform: 
  // https://www.terraform.io/docs/providers/google/r/compute_instance.html#min_cpu_platform
  allow_stopping_for_update = true

  // Disk storage
  boot_disk {
    initialize_params {
      // TODO: is this `size` needed anymore? I originally set it for packer.. oops.
      size = "20" 
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
    scopes = ["userinfo-email", "compute-ro", "storage-ro"]
  }

  # ------------------------------------------------
  # GPU Configuration options
  # ------------------------------------------------
  scheduling {
    # Instances with guest accelerators do not support live migration.
    on_host_maintenance = "TERMINATE"
  }
  guest_accelerator {
    count = 1
    type = "nvidia-tesla-p100"
  }
}
