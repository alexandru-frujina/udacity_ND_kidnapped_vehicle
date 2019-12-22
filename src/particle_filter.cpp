/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 30;  // TODO: Set the number of particles
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++)
  {
    Particle* particle = new Particle();

    particle->id = i;
    particle->x = dist_x(gen);
    particle->y = dist_y(gen);
    particle->theta = dist_theta(gen);
    particle->weight = 1.0;

    particles.push_back(*particle);

    std::cout << "theta " << particle->theta << std::endl;
  }

  srand(time(0));
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  std::normal_distribution<double> dist_x(0, std_x);
  std::normal_distribution<double> dist_y(0, std_y);
  std::normal_distribution<double> dist_theta(0, std_theta);
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++)
  {
    if (std::abs(yaw_rate) < 0.00001)
    {
      particles[i].x += velocity * delta_t * std::cos(particles[i].theta);
      particles[i].y += velocity * delta_t * std::sin(particles[i].theta);
      particles[i].theta += 0;
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (std::sin(particles[i].theta + yaw_rate * delta_t) - std::sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (std::cos(particles[i].theta) - std::cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned int i = 0; i < observations.size(); i++)
  {
    double myDist = -1;
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      double calc_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (myDist < 0 || calc_dist < myDist)
      {
        observations[i].id = predicted[j].id;
        myDist = calc_dist;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double std_landmark_x = std_landmark[0];
  double std_landmark_y = std_landmark[1];

  for (int i = 0; i < num_particles; i++)
  {
    vector<LandmarkObs> predicted;
    vector<LandmarkObs> mapObservations;
    double weightTemp = 1.0;

    // Create a new observations in map coordinates based on each particle
    for (unsigned int j = 0; j < observations.size(); j++)
    {
      LandmarkObs* landmarkObs = new LandmarkObs();
      landmarkObs->x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
      landmarkObs->y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);
      landmarkObs->id = 0;

      mapObservations.push_back(*landmarkObs);
    }

    // Make a list of the closest landmarks to a particle
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      double landmark_map_x = map_landmarks.landmark_list[j].x_f;
      double landmark_map_y = map_landmarks.landmark_list[j].y_f;

      if (dist(landmark_map_x, landmark_map_y, particles[i].x, particles[i].y) < sensor_range)
      {
        LandmarkObs* landmark = new LandmarkObs();
        landmark->id = map_landmarks.landmark_list[j].id_i;
        landmark->x = landmark_map_x;
        landmark->y = landmark_map_y;

        predicted.push_back(*landmark);
      }
    }

    // Get the associated closest landmarks of the current observations
    dataAssociation(predicted, mapObservations);

    for (unsigned int j = 0; j < mapObservations.size(); j++)
    {
      double mu_x = 0.0;
      double mu_y = 0.0;

      for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++)
      {
        if (map_landmarks.landmark_list[k].id_i == mapObservations[j].id)
        {
          mu_x = map_landmarks.landmark_list[k].x_f;
          mu_y = map_landmarks.landmark_list[k].y_f;

          break;
        }
      }

      double obs_x = mapObservations[j].x;
      double obs_y = mapObservations[j].y;

      double elem_x = (obs_x - mu_x) * (obs_x - mu_x) / (2 * std_landmark_x * std_landmark_x);
      double elem_y = (obs_y - mu_y) * (obs_y - mu_y) / (2 * std_landmark_y * std_landmark_y);

      weightTemp *= 1.0 / (2.0 * M_PI * std_landmark_x * std_landmark_y) * exp(-(elem_x + elem_y));
    }

    particles[i].weight = weightTemp;

    vector<int> myn_id;
    vector<double> myn_x;
    vector<double> myn_y;

    for (unsigned int j = 0; j < mapObservations.size(); j++)
    {
      myn_id.push_back(mapObservations[j].id);
      myn_x.push_back(mapObservations[j].x);
      myn_y.push_back(mapObservations[j].y);
    }

    mapObservations.clear();
    predicted.clear();

    SetAssociations(particles[i], myn_id, myn_x, myn_y);
  }
}

//#include <time.h>
void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  double beta = 0.0;
  int index;
  double wMax = 0;
  std::vector<Particle> newParticles;

  index = rand() % num_particles;

  for (int i = 0; i < num_particles; i++)
  {
    if (wMax < particles[i].weight)
    {
      wMax = particles[i].weight;
    }
  }

  for (int i = 0; i < num_particles; i++)
  {
    beta = beta + (double)rand() / RAND_MAX * 2 * wMax;

    while (particles[index].weight < beta)
    {
      beta = beta - particles[index].weight;
      index++;
      index = index % num_particles;
    }

    newParticles.push_back(particles[index]);
  }

  particles = newParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}