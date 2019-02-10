variable "image" {
  default = "mt-tesla-t4-img"
}

// Configure the Google Cloud provider
provider "google" {
  project     = "stop-harassing-staging"
  region      = "us-east1-c"
}

resource "google_compute_instance" "mathtastic-self-play-12" {
  count                     = "1"
  name                      = "mathtastic-12-30-t4"
  machine_type              = "custom-12-30720-ext"
  zone                      = "us-east1-c"
  metadata_startup_script   = "/usr/local/bin/monstertastic-boot-script.sh"
  tags                      = ["gpu-compute"]
  // Use an optimized Tensorflow build that from: https://github.com/mind/wheels/
  min_cpu_platform = "Intel Broadwell"
  // Stopping for update is required to set the cpu platform: 
  // https://www.terraform.io/docs/providers/google/r/compute_instance.html#min_cpu_platform
  allow_stopping_for_update = true

  // Disk storage
  boot_disk {
    initialize_params {
      image = "mt-tesla-t4-img"
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
  guest_accelerator {
    count = 1
    type = "nvidia-tesla-t4"
  }
}
